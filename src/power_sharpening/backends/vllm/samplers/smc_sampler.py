"""
Sequential Monte Carlo (SMC) power sampler based on VLLM.

Run with the installed package:
    python -m power_sharpening.runners.vllm.run_power_smc --dataset math500 \
        --override top_k=0 top_p=0.9

Reference: https://github.com/ArminAzizi98/Power-SMC

The proposal supports top-k and top-p filtering through SamplingParams or the sampler's keyword overrides. The runner
defaults to top_k=0 (disabled) and top_p=0.9, matching the upstream Power-SMC configuration. EOS is suppressed for the
first min_new_tokens generated tokens per particle (100 by default); this minimum is counted across generation chunks.

Targets pi(x) ~ p(x)^alpha using a tempered proposal q(x). One loop runs over `max_new_tokens` decoding steps; vLLM is
invoked in chunks of `block_size` tokens (vLLM cannot expose per-token control, so a chunk is the smallest call we can
make), the returned per-token logprobs drive a token-by-token weight update, and systematic resampling fires at every
block boundary when ESS < `ess_threshold * N`.

"""

import logging
import re
import time
from collections.abc import Mapping, Sequence
from typing import Optional

import torch

from vllm.inputs import TokensPrompt

from power_sharpening.backends.vllm.engine_patch import SamplingParams
from power_sharpening.backends.vllm.engine_patch.sampling_params import _copy_sampling_params

logger = logging.getLogger("[Power-SMC sampler]")


def _extract_logprobs(logprobs: list[dict]) -> list[float]:
    """Extract the top-1 logprob value from each decoding step."""
    return [list(lp.values())[0].logprob for lp in logprobs]


def _alpha_ramp(t: int, alpha_final: float, ramp_T: int) -> float:
    """Linearly anneal alpha from 1 to alpha_final over ramp_T tokens, then hold.

    Gradual ramping keeps consecutive SMC targets pi ~ p^alpha close, which avoids weight collapse on the first step.
    """
    if ramp_T <= 1:
        return float(alpha_final)
    if t < ramp_T:
        frac = float(t + 1) / float(ramp_T)
        return 1.0 + (float(alpha_final) - 1.0) * frac
    return float(alpha_final)


def _normalize_log_weights(log_w: torch.Tensor) -> torch.Tensor:
    """Normalize log-weights to probabilities via log-sum-exp."""
    lw = log_w - torch.max(log_w)
    w = torch.exp(lw)
    return w / torch.sum(w)


def _effective_sample_size(log_w: torch.Tensor) -> float:
    """ESS = 1 / sum(w_i^2) over normalized weights."""
    w = _normalize_log_weights(log_w)
    return float(1.0 / torch.sum(w * w))


def _systematic_resample(
    log_w: torch.Tensor, rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Systematic resampling. Returns N ancestor indices ~ Cat(w).

    One u0 ~ U(0, 1) defines the deterministic comb {(u0 + i) / N}, and particle j is selected for every comb position
    falling in its CDF slot [F_{j-1}, F_j). Guarantees each particle gets floor(N*w_j) or ceil(N*w_j) copies — strictly
    lower variance than multinomial.
    """
    w = _normalize_log_weights(log_w)
    N = w.numel()
    device = w.device
    u0 = (
        torch.rand((), device=device, generator=rng)
        if rng is not None
        else torch.rand((), device=device)
    )
    positions = (u0 + torch.arange(N, device=device)) / N
    cdf = torch.cumsum(w, dim=0)
    cdf[-1] = 1.0
    idx = torch.searchsorted(cdf, positions, right=False)
    return idx.clamp_max(N - 1).to(torch.long)


def _resolve_device(device: "str | torch.device | None") -> torch.device:
    """Device SMC bookkeeping tensors should live on.

    vLLM manages the model's own device placement internally and doesn't expose it on the wrapper, so this mirrors
    vLLM's own default (the process's current CUDA device) rather than introspecting the engine.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda")


_BOX_ONLY_RE = re.compile(r"^[\s{}]*$")


def _has_nonempty_boxed(text: str) -> bool:
    """True if `text` contains a `\\boxed{...}`/`\\fbox{...}` with non-empty content.

    Ported verbatim from `smc_samp_utils.py`: finds the macro, then does brace-depth matching (not a fixed-width regex)
    since the content can itself contain braces.
    """
    for macro in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(macro, start)
            if idx < 0:
                break
            j = idx + len(macro)
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] == "{":
                depth = 0
                right = None
                for k in range(j, len(text)):
                    c = text[k]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            right = k
                            break
                if right is not None:
                    content = text[j + 1 : right].strip()
                    while (
                        len(content) >= 2 and content[0] == "{" and content[-1] == "}"
                    ):
                        inner = content[1:-1].strip()
                        if inner == content:
                            break
                        content = inner
                    if content and (_BOX_ONLY_RE.fullmatch(content) is None):
                        return True
            start = idx + len(macro)
    return False


def smc_power_sampler(
    mh_llm_model,
    prompts,
    sampling_params: SamplingParams,
    max_new_tokens: int = 3_072,
    block_size: int = 128,
    n_particles: int = 32,
    ess_threshold: float = 0.5,
    alpha_ramp_tokens: int = 400,
    temperature_scheduler=None,
    stop_on_boxed: bool = True,
    boxed_check_window_tokens: int = 256,
    device: "str | torch.device | None" = None,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
    min_new_tokens: int = 100,
) -> list[str]:
    """Generate text using SMC power sampling targeting pi(x) ~ p(x)^alpha.

    Faithful port of `smc_power_sample` in `algorithms/Power-SMC/smc_samp_utils.py` to a vLLM backend. The original runs
    a single per-token loop for `max_new_tokens` steps and only consults `block_size` to decide when to resample (`(t +
    1) % block_size == 0`). vLLM does not expose per-token control, so we ask it for `block_size` tokens per call and
    unroll the per-token weight update over the returned logprobs.

    Args:
        mh_llm_model:    vLLM-backed model wrapper with `.llm` and `.tokenizer`.
        prompts:         One prompt or a list of prompts. Each prompt may be a
                         rendered string, a vLLM token-prompt object, or a list of token ids. Strings are tokenized once
                         before SMC.
        sampling_params: Sampling configuration; `sampling_params.alpha`
                         sets the target power.
        max_new_tokens:  Total tokens generated per particle (the outer
                         loop's horizon). Matches `cfg.max_new_tokens`.
        block_size:      Resampling cadence in tokens. Matches
                         `cfg.block_size`.
        n_particles:     Number of SMC particles per prompt.
        ess_threshold:   Resample when ESS < ess_threshold * n_particles.
        alpha_ramp_tokens: Tokens over which alpha ramps from 1 to alpha.
        temperature_scheduler: Optional schedule producing a temperature
                         per block (called once per vLLM chunk).
        stop_on_boxed:   Mark a particle done as soon as a non-empty
                         `\\boxed{...}`/`\\fbox{...}` appears in its recent output, matching `cfg.stop_on_boxed` in the
                         reference (always enabled there).
        boxed_check_window_tokens: Only decode/check the last this-many
                         generated tokens per particle per step. Matches `cfg.boxed_check_window_tokens`.
        device:          Device for the SMC bookkeeping tensors (log_w,
                         cum_logp, done, ...). Defaults to the current CUDA device if available, else CPU.
        top_k:           Override the proposal's top-k filter; 0 or -1 disables
                         it. None preserves sampling_params.top_k.
        top_p:           Override the proposal's nucleus probability in (0, 1];
                         1 disables it. None preserves sampling_params.top_p.
        min_new_tokens:  Suppress EOS/stop-token IDs for this many generated
                         tokens per particle, counted across all chunks. Defaults to the upstream minimum of 100; 0
                         disables suppression. Boxed-answer stopping remains independent.

    Returns:
        A list of decoded strings (one per prompt).
    """
    # Normalise to list so the rest of the method has a single code-path.
    if isinstance(prompts, str):
        prompts = [prompts]

    if not isinstance(min_new_tokens, int) or min_new_tokens < 0:
        raise ValueError("min_new_tokens must be a non-negative integer")

    # Only the sampled token's scores are needed for SMC. With logprobs=0, both streams contain that same token even
    # when EOS masking changes their top-ranked alternatives.
    sampling_params = _copy_sampling_params(
        sampling_params, n=1, logprobs=0, logprob_token_ids=None
    )
    # Apply filters inside vLLM so out.logprobs describes the filtered proposal q. out.power_logprobs still describes
    # the unfiltered base model p. Every subsequent chunk clones these settings without changing the caller.
    if top_k is not None:
        if not isinstance(top_k, int) or top_k < -1:
            raise ValueError("top_k must be -1, 0, or a positive integer")
        sampling_params.top_k = top_k
    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        sampling_params.top_p = top_p
    alpha_final = float(sampling_params.alpha)
    tokenizer = mh_llm_model.tokenizer

    prompt_token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]

    ramp_T = max(int(alpha_ramp_tokens), 1)
    eos_token_id = tokenizer.eos_token_id
    device = _resolve_device(device)
    rng = torch.Generator(device=device)

    logger.debug(
        "n_prompts=%d, alpha=%.4f, n_particles=%d, block_size=%d, "
        "max_new_tokens=%d, min_new_tokens=%d, ess_thr=%.2f, ramp=%d, "
        "top_k=%d, top_p=%.4f",
        len(prompt_token_ids),
        alpha_final,
        n_particles,
        block_size,
        max_new_tokens,
        min_new_tokens,
        ess_threshold,
        ramp_T,
        sampling_params.top_k,
        sampling_params.top_p,
    )

    P = len(prompt_token_ids)
    N = n_particles

    # Per-(prompt, particle) state. Shapes: (P, N) for arrays; particles is [P][N][token_id], a nested list since rows
    # grow at different rates.
    particles: list[list[list[int]]] = [[[] for _ in range(N)] for _ in range(P)]
    log_w = torch.zeros((P, N), dtype=torch.float32, device=device)
    cum_logp = torch.zeros((P, N), dtype=torch.float32, device=device)
    done = torch.zeros((P, N), dtype=torch.bool, device=device)
    prev_alpha = 1.0
    resample_count = torch.zeros(P, dtype=torch.int64, device=device)
    ess_history: list[list[float]] = [[] for _ in range(P)]

    logger.info(
        "starting SMC: n_prompts=%d, n_particles=%d (batch=%d/chunk)",
        P,
        N,
        P * N,
    )

    run_start = time.perf_counter()
    time_build = time_forward = time_collect = time_update = 0.0
    time_decode = 0.0

    # --- Single per-token loop, exactly like the original. ---------------
    # vLLM is invoked in chunks of `block_size` because it cannot generate a single token at a time without losing
    # efficiency. We now submit every (prompt, particle) pair as one batch per chunk, so one vLLM call advances all
    # prompts in lockstep.
    t = 0
    while t < max_new_tokens:
        if done.all():
            logger.info("all particles done at step %d", t)
            break

        chunk_size = min(block_size, max_new_tokens - t)

        # Per-chunk temperature schedule (mirrors power_sampling_mh.py). Single global schedule across all prompts in
        # this batched version.
        if temperature_scheduler is not None:
            temperature_scheduler.step_count = 0
            blk_temp = temperature_scheduler.step()
            sampling_params = _copy_sampling_params(
                sampling_params, temperature=blk_temp
            )
            logger.info("step=%d: set temperature %.6f", t, blk_temp)

        build_start = time.perf_counter()
        # Flatten active (prompt, particle) pairs into a single batch.
        active_pairs = [(p, i) for p in range(P) for i in range(N) if not done[p, i]]
        context_token_ids = {
            (p, i): prompt_token_ids[p] + particles[p][i] for (p, i) in active_pairs
        }
        fwd_prompts = [
            TokensPrompt(prompt_token_ids=context_token_ids[(p, i)])
            for (p, i) in active_pairs
        ]
        blk_build = time.perf_counter() - build_start
        time_build += blk_build

        logger.info(
            "step=%d/%d: active=%d/%d, chunk=%d",
            t,
            max_new_tokens,
            len(active_pairs),
            P * N,
            chunk_size,
        )
        # vLLM counts min_tokens from the start of each fresh request. Carry only the remaining global minimum into each
        # particle's next chunk, capped by this request's generation budget (which may be shorter).
        # A per-request seed fixes that request's sampled tokens, so one shared seed would make particles with the same
        # context (all of them at t=0, and every copy made by resampling) generate identical continuations. Give each
        # (chunk, prompt, particle) request its own seed derived from the run seed.
        base_seed = sampling_params.seed
        fwd_sp = [
            _copy_sampling_params(
                sampling_params,
                max_tokens=chunk_size,
                min_tokens=min(
                    chunk_size,
                    max(0, min_new_tokens - len(particles[p][i])),
                ),
                seed=(
                    None
                    if base_seed is None
                    else (base_seed + (t * P + p) * N + i) % (2**63)
                ),
            )
            for p, i in active_pairs
        ]
        forward_start = time.perf_counter()
        fwd_outputs = mh_llm_model.llm.generate(
            fwd_prompts, sampling_params=fwd_sp, use_tqdm=False
        )
        blk_forward = time.perf_counter() - forward_start
        time_forward += blk_forward

        # Collect chunk tokens and per-token logp / logq, keyed by (p, i).
        #   logp := base-model logprob  (target's per-token loglik)
        #   logq := proposal (tempered) logprob actually sampled from
        chunk_tokens: dict[tuple[int, int], list[int]] = {}
        chunk_logp: dict[tuple[int, int], list[float]] = {}
        chunk_logq: dict[tuple[int, int], list[float]] = {}
        mismatched_logprob_lengths: list[tuple[int, int, int, int, int, int]] = []
        raw_logprob_examples: list[dict[str, object]] = []
        collect_start = time.perf_counter()
        for batch_idx, (p, i) in enumerate(active_pairs):
            out = fwd_outputs[batch_idx].outputs[0]
            token_ids = list(out.token_ids)
            raw_logprobs_len = len(out.logprobs)
            raw_power_logprobs_len = len(out.power_logprobs)
            # vLLM's processed logprobs are the proposal q; our custom power_logprobs are pre-temperature/base-model log
            # p.
            proposal_logq = _extract_logprobs(out.logprobs)
            base_logp = _extract_logprobs(out.power_logprobs)
            aligned_len = min(len(token_ids), len(base_logp), len(proposal_logq))
            if aligned_len < len(token_ids):
                mismatched_logprob_lengths.append(
                    (
                        p,
                        i,
                        len(token_ids),
                        len(base_logp),
                        len(proposal_logq),
                        aligned_len,
                    )
                )
                if len(raw_logprob_examples) < 5:
                    raw_logprob_examples.append(
                        {
                            "p": p,
                            "i": i,
                            "output_type": type(out).__name__,
                            "finish_reason": getattr(out, "finish_reason", None),
                            "stop_reason": getattr(out, "stop_reason", None),
                            "raw_tokens": len(token_ids),
                            "raw_logprobs": raw_logprobs_len,
                            "raw_power_logprobs": raw_power_logprobs_len,
                            "extracted_logq": len(proposal_logq),
                            "extracted_logp": len(base_logp),
                            "aligned": aligned_len,
                            "has_power_attr": hasattr(out, "power_logprobs"),
                        }
                    )
            chunk_tokens[(p, i)] = token_ids[:aligned_len]
            chunk_logp[(p, i)] = base_logp[:aligned_len]
            chunk_logq[(p, i)] = proposal_logq[:aligned_len]
        blk_collect = time.perf_counter() - collect_start
        time_collect += blk_collect

        align_start = time.perf_counter()
        for p, i in active_pairs:
            aligned_len = min(
                len(chunk_tokens[(p, i)]),
                len(chunk_logp[(p, i)]),
                len(chunk_logq[(p, i)]),
            )
            if aligned_len < len(chunk_tokens[(p, i)]):
                mismatched_logprob_lengths.append(
                    (
                        p,
                        i,
                        len(chunk_tokens[(p, i)]),
                        len(chunk_logp[(p, i)]),
                        len(chunk_logq[(p, i)]),
                        aligned_len,
                    )
                )
            chunk_tokens[(p, i)] = chunk_tokens[(p, i)][:aligned_len]
            chunk_logp[(p, i)] = chunk_logp[(p, i)][:aligned_len]
            chunk_logq[(p, i)] = chunk_logq[(p, i)][:aligned_len]
        blk_align = time.perf_counter() - align_start
        blk_collect += blk_align
        time_collect += blk_align

        if mismatched_logprob_lengths:
            logger.warning(
                "step=%d: truncated %d/%d particles to aligned token/logprob "
                "prefixes; first examples=(p,i,tokens,logp,logq,aligned) %s",
                t,
                len(mismatched_logprob_lengths),
                len(active_pairs),
                mismatched_logprob_lengths[:5],
            )
            logger.warning(
                "step=%d raw vLLM output diagnostics: %s",
                t,
                raw_logprob_examples,
            )

        # ---- Per-token loop inside the chunk (matches the original) -----
        # At every position k of the chunk we have a "global" step
        # t_global = t + k. We do:
        #   - alpha ramp (with retroactive delta * cum_logp correction)
        #   - IS weight update (alpha_t-1)*logp + (logp - logq)
        #   - per-particle EOS / done detection (token-level)
        #   - resample check exactly when (t_global + 1) % block_size == 0 or t_global + 1 == max_new_tokens. Resampling
        #     stays scoped to each prompt's own N particles.
        chunk_len = max((len(chunk_tokens[key]) for key in active_pairs), default=0)
        if active_pairs and chunk_len == 0:
            examples = [
                (
                    p,
                    i,
                    len(chunk_tokens[(p, i)]),
                    len(chunk_logp[(p, i)]),
                    len(chunk_logq[(p, i)]),
                )
                for p, i in active_pairs[:5]
            ]
            raise RuntimeError(
                "vLLM returned no aligned token/logprob triples for the SMC "
                "chunk; cannot update particle weights. "
                "examples=(p,i,tokens,logp,logq) "
                f"{examples}. The custom backend must return both out.logprobs "
                "and out.power_logprobs for generated tokens. "
                f"raw_output_diagnostics={raw_logprob_examples}"
            )
        update_start = time.perf_counter()
        for k in range(chunk_len):
            t_global = t + k
            alpha_t = _alpha_ramp(t_global, alpha_final, ramp_T)

            # (a) alpha bumped: retroactive correction on all particles
            delta = alpha_t - prev_alpha
            if delta != 0.0:
                log_w = log_w + delta * cum_logp
                prev_alpha = alpha_t

            # (b) + (c): incremental weight for the new token x_t.
            for p, i in active_pairs:
                if done[p, i] or k >= len(chunk_tokens[(p, i)]):
                    continue
                lp = chunk_logp[(p, i)][k]
                lq = chunk_logq[(p, i)][k]
                log_w[p, i] += (alpha_t - 1.0) * lp + (lp - lq)
                cum_logp[p, i] += lp

                # commit the token and check EOS / length cap
                tok = chunk_tokens[(p, i)][k]
                particles[p][i].append(tok)
                if tok == eos_token_id or len(particles[p][i]) >= max_new_tokens:
                    done[p, i] = True
                elif stop_on_boxed:
                    window = particles[p][i][-boxed_check_window_tokens:]
                    gen_text_tail = tokenizer.decode(window, skip_special_tokens=True)
                    if _has_nonempty_boxed(gen_text_tail):
                        done[p, i] = True

            # Resample check at the per-token granularity, matching the original `is_block_end = ((t+1) % block_size ==
            # 0) or (t == Tmax-1)`.
            is_block_end = ((t_global + 1) % block_size == 0) or (
                t_global + 1 == max_new_tokens
            )
            if is_block_end:
                for p in range(P):
                    if done[p].all():
                        # This prompt's particles are all finished; the reference stops touching a prompt's weights the
                        # instant done.all() (it breaks before ever reaching the resample check). Resampling here would
                        # zero out the final importance weights right before they're used for the weighted particle
                        # choice.
                        continue
                    ess = _effective_sample_size(log_w[p])
                    ess_history[p].append(ess)
                    logger.info(
                        "prompt[%d] step=%d: ESS=%.2f/%d, mean_logw=%.3f, "
                        "max_logw=%.3f, done=%d/%d",
                        p,
                        t_global,
                        ess,
                        N,
                        float(torch.mean(log_w[p])),
                        float(torch.max(log_w[p])),
                        int(done[p].sum()),
                        N,
                    )
                    if ess < ess_threshold * N:
                        idx = _systematic_resample(log_w[p], rng=rng)
                        idx_list = idx.tolist()
                        particles[p] = [list(particles[p][j]) for j in idx_list]
                        cum_logp[p] = cum_logp[p][idx]
                        done[p] = done[p][idx]
                        log_w[p] = 0.0
                        resample_count[p] += 1
                        logger.info(
                            "prompt[%d] step=%d: RESAMPLED (total=%d)",
                            p,
                            t_global,
                            int(resample_count[p]),
                        )

            if done.all():
                break
        blk_update = time.perf_counter() - update_start
        time_update += blk_update
        logger.info(
            "step=%d timing: build=%.2fs, forward=%.2fs, collect=%.2fs, update=%.2fs",
            t,
            blk_build,
            blk_forward,
            blk_collect,
            blk_update,
        )

        # Advance global token counter by the actual chunk length.
        t += chunk_len if chunk_len > 0 else chunk_size

    # Final alpha catch-up so the weights reflect the full target.
    if prev_alpha != alpha_final:
        log_w = log_w + (alpha_final - prev_alpha) * cum_logp
        prev_alpha = alpha_final

    # Sample one chosen particle per prompt ~ normalized weights.
    decode_start = time.perf_counter()
    results: list[str] = []
    for p in range(P):
        probs = _normalize_log_weights(log_w[p])
        chosen_idx = int(torch.multinomial(probs, 1, generator=rng).item())
        decoded = tokenizer.decode(particles[p][chosen_idx], skip_special_tokens=True)
        results.append(decoded)

        avg_ess = (
            sum(ess_history[p]) / len(ess_history[p])
            if ess_history[p]
            else float("nan")
        )
        logger.info(
            "prompt[%d]: chosen particle=%d, gen_len=%d, resamples=%d, avg_ESS=%.2f",
            p,
            chosen_idx,
            len(particles[p][chosen_idx]),
            int(resample_count[p]),
            avg_ess,
        )
        logger.debug("prompt[%d]: final log_w=%s", p, log_w[p].tolist())
    time_decode = time.perf_counter() - decode_start

    total_time = time.perf_counter() - run_start
    logger.info(
        "SMC timing summary: total=%.2fs, build=%.2fs, forward=%.2fs, "
        "collect=%.2fs, update=%.2fs, decode=%.2fs",
        total_time,
        time_build,
        time_forward,
        time_collect,
        time_update,
        time_decode,
    )

    return results
