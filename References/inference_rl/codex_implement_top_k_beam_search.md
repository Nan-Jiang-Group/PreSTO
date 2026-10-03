# Top‑k‑Constrained Beam Search in vLLM (Design + Skeleton)

This document describes a practical way to implement Top‑k‑Constrained Beam Search (k‑CBS) using vLLM. It focuses on an implementation that:

- Uses vLLM’s `SamplingParams(logprobs=k, max_tokens=1)` to fetch the next‑token top‑k distribution per active beam.
- Maintains the beam state in Python (scores, token sequences, EOS state).
- Expands each active beam only with its top‑k tokens, renormalizes within the k‑set, and prunes globally to beam width B=k.
- Can run in a simple iterative `generate()` loop or an advanced “live engine” mode to reuse KV cache between steps.

---

## Quick Recap: k‑CBS

- Each step: for every active beam, take the per‑beam top‑k next tokens, renormalize within that k‑set, accumulate scores, then globally prune to top B (set B=k for k‑CBS).
- Variants: final‑k (return final B beams) and k² pre‑prune (collect up to k×k candidates at the last step before pruning for tighter bounds).

---

## API Shape (Proposed)

Add `reasoning-with-sampling/llm_experiments/topk_constrained_beam.py` with a function that takes an `LLM`, `tokenizer`, base prompt token ids, `beam_width` (set equal to `k`), `k`, and `max_new_tokens`.

```python
from typing import List, Dict, Optional, Tuple 
from dataclasses import dataclass
import math
from vllm import LLM, SamplingParams

@dataclass
class Beam:
    tokens: List[int]
    score: float
    ended: bool
    raw_score: float = 0.0

def logsumexp(vals: List[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))

# this explicitly looks at logprobs of the top 20 tokens then sorts them
# and decide which one to pick as topk
# so logsumexp is sued to renormalize the logprobs of top 20 and then sample accordingly
# to form a proper probability distribution.
# we could instead use vllm to sample topk?
def topk_constrained_beam_search(
    llm: LLM,
    tokenizer,
    base_prompt_ids: List[int],
    beam_width: int,
    k: int,
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    length_penalty: float = 1.0,
    return_k2_preprune: bool = False,
) -> Dict:
    assert beam_width >= 1 and k >= 1
    if eos_id is None:
        eos_id = tokenizer.eos_token_id

    beams = [Beam(tokens=[], score=0.0, ended=False, raw_score=0.0)]
    last_frontier = []

    for step in range(max_new_tokens):
        active = [b for b in beams if not b.ended]
        if not active:
            break

        prompts = [base_prompt_ids + b.tokens for b in active]
        sp = SamplingParams(
            max_tokens=1,
            logprobs=min(k, 10),
            temperature=1.0,
            top_k=0,
            top_p=1.0,
        )
        outs = llm.generate(prompts, sp)

        candidates: List[Beam] = []
        for b, out in zip(active, outs):
            item = out.outputs[0]
            top_logprobs = item.logprobs[0]  # dict-like
            pairs = []
            for tk, lp in top_logprobs.items():
                if isinstance(tk, str):
                    tid = tokenizer.convert_tokens_to_ids(tk)
                elif hasattr(tk, 'token_id'):
                    tid = tk.token_id
                else:
                    continue
                pairs.append((tid, float(lp)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = pairs[:k]

            lse = logsumexp([lp for _, lp in pairs]) if pairs else 0.0
            for tid, lp in pairs:
                new_tokens = b.tokens + [tid]
                ended = (tid == eos_id)
                renorm_lp = lp - lse
                new_score = b.score + renorm_lp
                if length_penalty != 1.0 and len(new_tokens) > 0:
                    new_score = new_score / (len(new_tokens) ** (length_penalty - 1.0))
                candidates.append(Beam(tokens=new_tokens, score=new_score, ended=ended))

        if return_k2_preprune and step == max_new_tokens - 1:
            last_frontier = candidates.copy()

        # merge duplicates
        best: Dict[Tuple[int, ...], Beam] = {}
        for c in candidates:
            key = tuple(c.tokens)
            if key not in best or c.score > best[key].score:
                best[key] = c
        beams = sorted(best.values(), key=lambda b: b.score, reverse=True)[:beam_width]

        if all(b.ended for b in beams):
            break

    return {"beams": beams, "k2_frontier": last_frontier if return_k2_preprune else None}
```

Notes:
- Many vLLM builds cap `SamplingParams.logprobs` (10–32 typical). Clamp `k` to the cap.
- We renormalize within the per‑beam top‑k set (`lp - logsumexp(topk_lp)`), matching top‑k decoding semantics.

---

## Vectorized Inner Loop (NumPy/Torch)

You can cut Python overhead by operating on arrays for the per‑step expansion. The heavy work is still model inference, but vectorizing the scoring/pruning helps, especially with larger `beam_width` and `k`.

NumPy example for one step (assuming `m` active beams and `k_eff = min(k, logprobs_cap)`):

```python
import numpy as np

# Build per-beam arrays from vLLM outputs
m = len(active)
k_eff = min(k, 10)  # or whatever cap you used in SamplingParams
token_ids = np.empty((m, k_eff), dtype=np.int32)
logps = np.empty((m, k_eff), dtype=np.float32)

for i, (b, out) in enumerate(zip(active, outs)):
    item = out.outputs[0]
    top_lp = item.logprobs[0]
    pairs = []
    for tk, lp in top_lp.items():
        if isinstance(tk, str):
            tid = tokenizer.convert_tokens_to_ids(tk)
        elif hasattr(tk, 'token_id'):
            tid = tk.token_id
        else:
            continue
        pairs.append((tid, float(lp)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:k_eff]
    token_ids[i, :len(pairs)] = [p[0] for p in pairs]
    logps[i, :len(pairs)] = [p[1] for p in pairs]

# Per-beam renormalization within the top-k set
row_max = logps.max(axis=1, keepdims=True)
lse = row_max + np.log(np.exp(logps - row_max).sum(axis=1, keepdims=True))
renorm = logps - lse  # shape (m, k_eff)

# Add base beam scores and optional length penalty
base_scores = np.array([b.score for b in active], dtype=np.float32)[:, None]
new_scores = base_scores + renorm

if length_penalty != 1.0:
    lens = np.array([len(b.tokens) + 1 for b in active], dtype=np.float32)[:, None]
    new_scores = new_scores / (lens ** (length_penalty - 1.0))

# Flatten and select top beam_width
flat_scores = new_scores.reshape(-1)
flat_idx = np.argpartition(-flat_scores, beam_width - 1)[:beam_width]
top_order = flat_idx[np.argsort(-flat_scores[flat_idx])]

# Materialize the top beams only (lazy sequence build)
new_beams = []
for idx in top_order:
    parent = idx // k_eff
    pos = idx % k_eff
    tid = int(token_ids[parent, pos])
    ended = (tid == tokenizer.eos_token_id)
    tokens = active[parent].tokens + [tid]
    score = float(flat_scores[idx])
    new_beams.append(Beam(tokens=tokens, score=score, ended=ended))

# Optional: de-dup the top beams (if needed)
unique = {}
for b in new_beams:
    key = tuple(b.tokens)
    if key not in unique or b.score > unique[key].score:
        unique[key] = b
beams = list(unique.values())[:beam_width]
```

Torch variant: mirror the above with `torch.tensor` ops and `torch.topk`. Since vLLM returns CPU floats, keeping this on CPU usually suffices; moving to GPU is rarely beneficial for these small arrays.

```python
import torch
logps_t = torch.as_tensor(logps)  # (m, k_eff)
lse_t = torch.logsumexp(logps_t, dim=1, keepdim=True)
renorm_t = logps_t - lse_t
base_t = torch.tensor([b.score for b in active]).unsqueeze(1)
new_scores_t = base_t + renorm_t
flat_scores_t = new_scores_t.flatten()
vals, idxs = torch.topk(flat_scores_t, k=beam_width)
# Proceed to build beams from `idxs` as in NumPy path
```

This style minimizes Python loops: only the final assembly of the top `beam_width` beams builds Python sequences.

---

## Simple Driver (One Prompt)

```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen2.5-Math-7B", trust_remote_code=True)
tok = llm.get_tokenizer()
base_ids = tok.encode("Solve: 2x+3=11. x?\n", add_special_tokens=False)
res = topk_constrained_beam_search(llm, tok, base_ids, beam_width=5, k=5, max_new_tokens=64)
for i, b in enumerate(res["beams"], 1):
    print(i, b.score, tok.decode(b.tokens, skip_special_tokens=True))
```

---

## Advanced “Live Engine” (KV Cache Reuse)

- Use `AsyncLLMEngine` (or `llm.llm_engine`) to keep requests alive.
- Per step: `add_request` with `prompt_token_ids`, `SamplingParams(max_tokens=1, logprobs=k)`, call `engine.step()`, collect top‑k logprobs, expand/prune beams, re‑issue updated requests.
- This avoids re‑encoding long prefixes each iteration and is faster for larger prompts/steps.

---

## Integration in This Repo

- Reuse `format_prompt` (`reasoning-with-sampling/llm_experiments/power_samp_utils.py`).
- Mirror `vllm_passk_math.py` to eval on MATH500 and save CSV compatible with `eval_vllm_passk.py`.
- Add a runner `run_vllm_kcbs.sh` like the existing beam/sampling scripts.

---

## Test Checklist

- Determinism across runs.
- Edge cases: early EOS, no EOS in top‑k, `k=1` (greedy), duplicates.
- Compare against vanilla beam (B=k) and best‑of‑N sampling to validate behavior and pass@k tradeoffs.
