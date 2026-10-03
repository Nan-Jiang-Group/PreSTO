import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import io
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
from vllm import LLM, SamplingParams
from tqdm import tqdm


@dataclass
class Beam:
    tokens: List[int]
    text: str
    score: float
    ended: bool
    raw_score: float = 0.0


def _logsumexp_np(x: np.ndarray, mask: Optional[np.ndarray] = None, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Numerically stable logsumexp with optional mask (0/1)."""
    if mask is not None:
        # Set invalid positions to -inf so they don't contribute
        x = np.where(mask.astype(bool), x, -1e30)
    m = np.max(x, axis=axis, keepdims=True)
    s = np.exp(x - m)
    if mask is not None:
        s = s * mask
    lse = m + np.log(np.sum(s, axis=axis, keepdims=True))
    if not keepdims:
        lse = np.squeeze(lse, axis=axis)
    return lse


def _extract_topk_candidates(top_entry, tokenizer, k_eff: int) -> List[Dict[str, object]]:
    """Return list of candidates with fields: {'tid': int|-1, 'text': str, 'lp': float}.

    Supports vLLM top-logprobs as:
      - list of objects with attributes token_id/logprob and optionally token
      - list of dicts with keys 'token_id'/'logprob' or 'token'/'logprob'
      - dict mapping token text -> logprob (float or object with .logprob)
    """
    cands: List[Dict[str, object]] = []
    try:
        if isinstance(top_entry, list):
            for obj in top_entry:
                try:
                    if hasattr(obj, "logprob"):
                        lp = float(getattr(obj, "logprob"))
                        tid = int(getattr(obj, "token_id")) if hasattr(obj, "token_id") else -1
                        # Prefer decoded_token if present, else token
                        if hasattr(obj, "decoded_token"):
                            text = str(getattr(obj, "decoded_token"))
                        else:
                            text = str(getattr(obj, "token")) if hasattr(obj, "token") else ""
                        cands.append({"tid": tid, "text": text, "lp": lp})
                    elif isinstance(obj, dict) and "logprob" in obj:
                        lp = float(obj["logprob"])  # type: ignore
                        tid = int(obj.get("token_id", -1))  # type: ignore
                        text = str(obj.get("token", ""))  # type: ignore
                        cands.append({"tid": tid, "text": text, "lp": lp})
                except Exception:
                    continue
        elif isinstance(top_entry, dict):
            for tk, val in top_entry.items():
                try:
                    # extract logprob
                    if isinstance(val, (float, int)):
                        lp = float(val)
                    elif hasattr(val, "logprob"):
                        lp = float(getattr(val, "logprob"))
                    elif isinstance(val, dict) and "logprob" in val:
                        lp = float(val["logprob"])  # type: ignore
                    else:
                        continue

                    # key may be token id (int) or decoded token (str)
                    tid = -1
                    text = ""
                    if isinstance(tk, int):
                        tid = int(tk)
                        try:
                            text = tokenizer.decode([tid], skip_special_tokens=True)
                        except Exception:
                            text = ""
                    else:
                        # assume decoded token string
                        text = str(tk)
                        try:
                            mapped = tokenizer.convert_tokens_to_ids(text)
                            if mapped is not None:
                                tid = int(mapped)
                        except Exception:
                            tid = -1

                    cands.append({"tid": tid, "text": text, "lp": lp})
                except Exception:
                    continue
    except Exception:
        pass

    # Sort desc by logprob and clip
    cands.sort(key=lambda d: d.get("lp", -1e30), reverse=True)
    return cands[:k_eff]


def _get_context_window(llm, tokenizer) -> int:
    """Best-effort retrieval of model context length."""
    # Try vLLM engine (if accessible)
    try:
        eng = getattr(llm, "llm_engine", None)
        if eng is not None and hasattr(eng, "model_config"):
            cw = getattr(eng.model_config, "max_model_len", None)
            if isinstance(cw, int) and cw > 0:
                return int(cw)
    except Exception:
        pass
    # Fallback to tokenizer
    cw = getattr(tokenizer, "model_max_length", 4096)
    try:
        cw_int = int(cw)
        if cw_int <= 0 or cw_int > 10**7:
            return 4096
        return cw_int
    except Exception:
        return 4096


def topk_constrained_beam_search(
    llm: LLM,
    tokenizer,
    base_prompt: str,
    *,
    beam_width: int,
    topk: int,
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    length_penalty: float = 1.0,
    return_k2_preprune: bool = False,
    debug: bool = False,
    suppress_lib_stdout: bool = False,
) -> Dict:
    """Run Top‑k‑Constrained Beam Search (k‑CBS) using vLLM logprobs.

    Args:
        llm: vLLM LLM instance
        tokenizer: tokenizer compatible with the LLM
        base_prompt: formatted input prompt string
        beam_width: number of sequences to keep (B). In k‑CBS, typically B = topk.
        topk: per-step top‑k next-token logprobs to consider per beam (top‑k mask)
        max_new_tokens: maximum tokens to generate
        eos_id: EOS token id (defaults to tokenizer.eos_token_id)
        length_penalty: >1 favors shorter, 1 means no penalty
        return_k2_preprune: if True, also return the last step frontier (k²)
        logprobs_cap: cap for vLLM logprobs (common caps ~10)

    Returns:
        dict with keys:
          - "beams": final top beams (list[Beam])
          - "k2_frontier": optional list[Beam] with last-step candidates
          - "k_eff": effective k used per step (min(k, logprobs_cap))
    """
    assert beam_width >= 1 and topk >= 1
    if eos_id is None:
        eos_id = tokenizer.eos_token_id

    topk_eff = int(topk)

    beams: List[Beam] = [Beam(tokens=[], text="", score=0.0, ended=False, raw_score=0.0)]
    last_frontier: List[Beam] = []

    # Respect model context window per-beam by checking tokenized length each step
    base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
    context_window = _get_context_window(llm, tokenizer)

    # safety margin accounts for tokenizer/template differences inside vLLM
    safety_margin = 4

    pbar = tqdm(total=max_new_tokens, desc="kCBS steps", leave=False) if debug else None
    for step in range(max_new_tokens):
        active = [b for b in beams if not b.ended]
        if not active:
            break

        # Build prompts for active beams by appending their current continuation
        prompts: List[str] = []
        kept_idx: List[int] = []
        for idx_b, b in enumerate(active):
            full_prompt = base_prompt + (b.text or "")
            # Guard context: disallow requesting if current tokens near limit
            try:
                cur_len = len(tokenizer.encode(full_prompt, add_special_tokens=False))
            except Exception:
                cur_len = len(base_ids) + len(b.tokens)
            # vLLM tends to count an extra position; be conservative
            if cur_len >= context_window - safety_margin:
                # mark ended; skip expansion
                b.ended = True
                continue
            prompts.append(full_prompt)
            kept_idx.append(idx_b)

        # Query one token ahead with per-beam top-k logprobs
        sp = SamplingParams(
            max_tokens=1,
            logprobs=topk_eff,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
        )
        if not prompts:
            # no expandable beams remain
            break
        if suppress_lib_stdout:
            _buf_out, _buf_err = io.StringIO(), io.StringIO()
            with redirect_stdout(_buf_out), redirect_stderr(_buf_err):
                outs = llm.generate(prompts, sp)
        else:
            outs = llm.generate(prompts, sp)

        m = len(prompts)
        token_ids = np.full((m, topk_eff), fill_value=-1, dtype=np.int32)
        token_texts: List[List[str]] = [[""] * topk_eff for _ in range(m)]
        logps = np.full((m, topk_eff), fill_value=-1e30, dtype=np.float32)
        valid_mask = np.zeros((m, topk_eff), dtype=np.float32)

        for i, out in enumerate(outs):
            item = out.outputs[0]
            # For max_tokens=1, item.logprobs is length-1 list
            top_entry = item.logprobs[0] if isinstance(item.logprobs, list) else item.logprobs
            cands = _extract_topk_candidates(top_entry, tokenizer, topk_eff)
            if not cands:
                continue
            r = len(cands)
            # Fill arrays
            for j, c in enumerate(cands):
                tid = int(c.get("tid", -1))
                token_ids[i, j] = tid
                token_texts[i][j] = str(c.get("text", ""))
                logps[i, j] = float(c.get("lp", -1e30))
                valid_mask[i, j] = 1.0

        # Renormalize within per-beam top-k set (mask invalid)
        lse = _logsumexp_np(logps, mask=valid_mask, axis=1, keepdims=True)
        renorm = logps - lse  # shape (m, k_eff)

        # Base scores correspond only to beams that were kept (prompts built)
        base_scores = np.asarray([active[i].score for i in kept_idx], dtype=np.float32)[:, None]
        new_scores = base_scores + renorm

        if length_penalty != 1.0:
            lens = np.asarray([len(active[i].tokens) + 1 for i in kept_idx], dtype=np.float32)[:, None]
            new_scores = new_scores / (lens ** (length_penalty - 1.0))

        # Keep optional k^2 frontier before pruning
        if return_k2_preprune and step == max_new_tokens - 1:
            # Materialize all candidates (valid only)
            frontier: List[Beam] = []
            for i in range(m):
                for j in range(topk_eff):
                    if valid_mask[i, j] <= 0:
                        continue
                    tid = int(token_ids[i, j]) if token_ids[i, j] >= 0 else -1
                    ended = (tid == eos_id) if tid >= 0 else False
                    parent_beam = active[kept_idx[i]]
                    tokens = parent_beam.tokens + ([tid] if tid >= 0 else [])
                    text_piece = token_texts[i][j] if token_texts[i][j] else (tokenizer.decode([tid], skip_special_tokens=True) if tid >= 0 else "")
                    text = (parent_beam.text or "") + text_piece
                    score = float(new_scores[i, j])
                    frontier.append(Beam(tokens=tokens, text=text, score=score, ended=ended))
            last_frontier = frontier

        # Global prune to top `beam_width`
        flat_scores = new_scores.reshape(-1)
        # Only consider valid candidate positions
        valid_flat_mask = valid_mask.reshape(-1) > 0
        valid_indices = np.nonzero(valid_flat_mask)[0]
        if valid_indices.size == 0:
            break
        # Select top-k among valid indices
        candidate_scores = flat_scores[valid_indices]
        if candidate_scores.size <= beam_width:
            top_rel_idx = np.arange(candidate_scores.size)
        else:
            part = np.argpartition(-candidate_scores, beam_width - 1)[:beam_width]
            top_rel_idx = part[np.argsort(-candidate_scores[part])]
        top_idx = valid_indices[top_rel_idx]

        # Materialize new beams
        new_beams: List[Beam] = []
        for idx in top_idx.tolist():
            parent = idx // topk_eff
            pos = idx % topk_eff
            tid = int(token_ids[parent, pos])
            ended = (tid == eos_id) if tid >= 0 else False
            parent_beam = active[kept_idx[parent]]
            tokens = parent_beam.tokens + ([tid] if tid >= 0 else [])
            text_piece = token_texts[parent][pos] if token_texts[parent][pos] else (tokenizer.decode([tid], skip_special_tokens=True) if tid >= 0 else "")
            text = (parent_beam.text or "") + text_piece
            score = float(flat_scores[idx])
            new_beams.append(Beam(tokens=tokens, text=text, score=score, ended=ended))

        # Optional de-dup
        merged_by_text: Dict[str, Beam] = {}
        for b in new_beams:
            key = b.text
            if key not in merged_by_text or b.score > merged_by_text[key].score:
                merged_by_text[key] = b

        beams = sorted(merged_by_text.values(), key=lambda x: x.score, reverse=True)[:beam_width]

        # Stop early if all ended
        if all(b.ended for b in beams):
            break

        if pbar is not None:
            try:
                pbar.update(1)
                pbar.set_postfix(active=len(active), prompts=len(prompts), kept=len(kept_idx))
            except Exception:
                pass

    return {
        "beams": beams,
        "k2_frontier": last_frontier if return_k2_preprune else None,
        "topk_eff": topk_eff,
    }
