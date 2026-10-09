import os
import time
import json
import tempfile
import asyncio
import httpx
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from openai import OpenAI, AsyncOpenAI
from datasets import load_dataset
import random
import re
import numpy as np

try:
    from async_openai_client import AsyncOpenAIClient, ChatCompletionResponse
except ImportError:
    from src.async_openai_client import AsyncOpenAIClient, ChatCompletionResponse


@dataclass
class SamplingResult:
    """Results for a single problem with a specific sampling strategy."""
    task_id: str
    completion: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_seconds: float
    passed: bool = False
    metadata: Optional[Dict] = None  # For benchmark-specific data

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "completion": self.completion,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "time_seconds": self.time_seconds,
            "passed": self.passed,
            "metadata": self.metadata or {}
        }


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a model/strategy combination."""
    model_name: str
    strategy_name: str
    benchmark_name: str
    pass_rate: float
    avg_time: float
    total_tokens: int
    avg_tokens_per_problem: float
    num_problems: int


class SamplingStrategy:
    """Base class for sampling strategies."""

    # Optional tokenizer for apply_chat_template - set via set_tokenizer()
    _tokenizer = None
    _tokenizer_model_name = None  # Track which model the tokenizer is for

    @classmethod
    def set_tokenizer(cls, tokenizer):
        """
        Set the tokenizer to use for chat template formatting.

        Usage:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
            SamplingStrategy.set_tokenizer(tokenizer)
        """
        cls._tokenizer = tokenizer

    @classmethod
    def set_tokenizer_from_model(cls, model_name: str, trust_remote_code: bool = True):
        """
        Automatically load tokenizer from a HuggingFace model name.

        This is called automatically when using MCMC/BeamSearch sampling strategies.

        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            trust_remote_code: Whether to trust remote code for custom tokenizers
        """
        # Skip if we already have a tokenizer for this model
        if cls._tokenizer is not None and cls._tokenizer_model_name == model_name:
            return

        try:
            from transformers import AutoTokenizer
            cls._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            cls._tokenizer_model_name = model_name
            print(f"[SamplingStrategy] Loaded tokenizer for {model_name}")
        except Exception as e:
            print(f"[SamplingStrategy] Warning: Could not load tokenizer for {model_name}: {e}")
            print(f"[SamplingStrategy] You may need to manually call SamplingStrategy.set_tokenizer()")
            raise

    def __init__(self, name: str, seed: int = None):
        self.name = name
        self.seed = seed

    def _apply_chat_template(self, prompt: str, prefix: str = "", model_name: str = None) -> str:
        """
        Apply chat template for raw completions API.

        Uses tokenizer.apply_chat_template() - tokenizer must be set via set_tokenizer().

        For continuations, prefix is appended after the generation prompt.
        NO closing tag - model continues from exactly where prefix ends.
        """
        return prompt + prefix
        # Auto-load tokenizer if model_name provided and no tokenizer set
        if self._tokenizer is None and model_name:
            self.set_tokenizer_from_model(model_name)

        if self._tokenizer is None:
            raise RuntimeError(
                "No tokenizer set. Call SamplingStrategy.set_tokenizer_from_model(model_name) first.\n"
                "Example:\n"
                "  SamplingStrategy.set_tokenizer_from_model('meta-llama/Llama-3.1-8B-Instruct')\n"
                "Or manually:\n"
                "  from transformers import AutoTokenizer\n"
                "  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')\n"
                "  SamplingStrategy.set_tokenizer(tokenizer)"
            )
        # Use tokenizer's chat template
        formatted = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        # Append prefix for continuations (no closing tag)
        return formatted + prefix

    def _is_safe_boundary(self, text: str) -> bool:
        """
        Check if text can be tokenized and detokenized without drift.

        Tokenizers are not perfectly reversible. When we slice tokens and rejoin
        them as a string, then send to the API, the API may re-tokenize differently.
        This validates that encode(decode(encode(text))) == encode(text).

        Returns True if the boundary is safe (no tokenization drift).
        """
        if self._tokenizer is None:
            return True  # Can't validate without tokenizer, assume safe

        try:
            # Encode the text
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)
            # Decode back to text
            decoded = self._tokenizer.decode(token_ids)
            # Re-encode
            re_encoded = self._tokenizer.encode(decoded, add_special_tokens=False)
            # Check if round-trip is lossless
            return token_ids == re_encoded
        except Exception:
            return True  # On error, assume safe to avoid blocking

    def _find_safe_prefix(self, tokens: list[str], target_idx: int, block_size: int = 16) -> tuple[int, str]:
        """
        Find a safe prefix boundary near target_idx where tokenization is reversible.

        Searches for a boundary where joining tokens produces text that can be
        re-tokenized to the same token IDs. This prevents tokenization drift
        that would invalidate the Metropolis-Hastings acceptance ratio.

        Args:
            tokens: List of token strings from the API
            target_idx: Target index to slice at
            block_size: Block size for searching (search within this range)

        Returns:
            (safe_idx, prefix_text) - The safe index and the prefix text to use
        """
        # Try the target index first
        prefix = "".join(tokens[:target_idx])
        if self._is_safe_boundary(prefix):
            return target_idx, prefix

        # Search for safe boundary near target
        for offset in range(1, min(block_size, target_idx)):
            # Try before target
            if target_idx - offset >= 1:
                test_prefix = "".join(tokens[:target_idx - offset])
                if self._is_safe_boundary(test_prefix):
                    return target_idx - offset, test_prefix

            # Try after target
            if target_idx + offset < len(tokens):
                test_prefix = "".join(tokens[:target_idx + offset])
                if self._is_safe_boundary(test_prefix):
                    return target_idx + offset, test_prefix

        # Fallback: use target anyway (may have drift, but better than nothing)
        return target_idx, prefix

    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        """
        Generate completion using this strategy.
        Returns: (completion, prompt_tokens, completion_tokens)
        """
        raise NotImplementedError


class GreedySampling(SamplingStrategy):
    """Greedy decoding with temperature=0 using chat completions API."""

    def __init__(self, seed: int = None):
        super().__init__("Greedy", seed=seed)

    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        response = client.chat.completions.create(
            model=client.default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            seed=self.seed,
        )
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )


class MCMCSampling(SamplingStrategy):
    """
    MCMC power sampling with Metropolis-Hastings acceptance and partial regeneration.

    Implements sampling from target π(x) = p(x)^α using proposal q(x) = p(x).
    Based on the paper "Reasoning with Sampling" (https://arxiv.org/abs/2510.14901).

    Algorithm:
    - Target distribution: π(x) = p(x)^α where α is specified
    - Proposal distribution: q(x) = p(x) (base model, temperature=1)
    - Partial regeneration: pick random position, regenerate suffix
    - Accept/reject using MH ratio on the suffix

    For α=4: proposals with higher log probability are 3x more likely to be accepted.
    """

    def __init__(
        self,
        alpha: float = 4.0,
        mcmc_steps: int = 10,
        top_logprobs: int = 5,
        proposal_temperature: float = 1.0,
        temperature: float = None,  # Legacy alias for proposal_temperature
        restrict_to_last_n: int = None,  # Only resample last N blocks (None = disabled)
        block_size: int = 192,  # Block size B for block-wise generation (paper default)
        debug: bool = False,  # Print debug info during MCMC
        seed: int = None,  # Random seed for API reproducibility
    ):
        name = f"MCMC(α={alpha},steps={mcmc_steps},B={block_size})"
        if restrict_to_last_n is not None:
            name += f",lastN={restrict_to_last_n}"
        super().__init__(name, seed=seed)
        self.alpha = alpha
        self.mcmc_steps = mcmc_steps
        self.top_logprobs = top_logprobs
        # Support legacy 'temperature' parameter as alias for proposal_temperature
        self.proposal_temperature = temperature if temperature is not None else proposal_temperature
        self.restrict_to_last_n = restrict_to_last_n
        self.block_size = block_size
        self.debug = debug

    def _extract_logprobs_completion(self, response) -> tuple[list[str], list[float], list[float]]:
        """
        Extract tokens and logprobs from completions API response.

        Completions API uses different structure than chat API:
        - response.choices[0].logprobs.tokens (list of strings)
        - response.choices[0].logprobs.token_logprobs (list of floats, first may be None)

        Returns:
            (tokens, log_p, log_target)
        """
        choice = response.choices[0]
        if not choice.logprobs:
            return [], [], []

        tokens = choice.logprobs.tokens
        log_p_raw = choice.logprobs.token_logprobs

        # Filter out None values (first token often has None logprob)
        valid = [(t, lp) for t, lp in zip(tokens, log_p_raw) if lp is not None]
        if not valid:
            return [], [], []

        tokens, log_p = zip(*valid)
        tokens, log_p = list(tokens), list(log_p)
        log_target = [self.alpha * lp for lp in log_p]

        return tokens, log_p, log_target

    def _sample_full(self, client: OpenAI, prompt: str, max_tokens: int):
        """Generate a full sample from base model using completions API."""
        full_prompt = self._apply_chat_template(prompt)  # No prefix for initial
        response = client.completions.create(
            model=client.default_model,
            prompt=full_prompt,
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
            seed=self.seed,
        )

        text = response.choices[0].text
        tokens, log_p, log_target = self._extract_logprobs_completion(response)
        # Track if completion ended naturally (EOS) vs hitting max_tokens
        finished_naturally = response.choices[0].finish_reason == "stop"

        return (
            text, tokens, log_p, log_target,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            finished_naturally
        )

    def _sample_continuation(self, client: OpenAI, prompt: str, prefix: str, max_tokens: int):
        """
        Generate a TRUE continuation from a prefix using completions API.

        Uses raw completions API with chat template - NO new assistant turn,
        just continues from exactly where the prefix ends.
        """
        full_prompt = self._apply_chat_template(prompt, prefix)  # Prefix appended, NO new turn
        response = client.completions.create(
            model=client.default_model,
            prompt=full_prompt,
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
        )

        continuation = response.choices[0].text  # TRUE continuation
        tokens, log_p, log_target = self._extract_logprobs_completion(response)
        finished_naturally = response.choices[0].finish_reason == "stop"

        return (
            continuation, tokens, log_p, log_target,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            finished_naturally
        )

    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        """
        Generate completion using MCMC power sampling with block-wise generation.

        Algorithm (matching paper):
        1. Generate tokens block-by-block (B tokens per block)
        2. After each block, run MCMC refinement steps
        3. MCMC uses block-aligned index selection (idx = block_idx * B)
        4. After all blocks, truncate at EOS if present
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0
        attempts = 0
        acceptances = 0

        # Initialize with empty generation
        tokens_cur = []
        log_p_cur = []
        log_target_cur = []

        # Calculate number of blocks to generate
        num_blocks_to_generate = max_tokens // self.block_size
        if num_blocks_to_generate < 1:
            num_blocks_to_generate = 1

        if self.debug:
            print(f"[MCMC] Block-wise generation: {num_blocks_to_generate} blocks of {self.block_size} tokens")

        # Generate block by block
        for block_num in range(num_blocks_to_generate):
            # Generate next block - validate tokenization boundary
            if tokens_cur:
                _, prefix = self._find_safe_prefix(tokens_cur, len(tokens_cur), self.block_size)
            else:
                prefix = ""

            if block_num == 0:
                # First block: use _sample_full (no prefix)
                block_text, block_tokens, block_log_p, block_log_target, pt, ct, _ = self._sample_full(
                    client, prompt, self.block_size
                )
            else:
                pass
                # Subsequent blocks: continue from prefix
                block_text, block_tokens, block_log_p, block_log_target, pt, ct, _ = self._sample_continuation(
                    client, prompt, prefix, self.block_size
                )

            total_prompt_tokens += pt
            total_completion_tokens += ct

            # Extend current state with new block
            tokens_cur.extend(block_tokens)
            log_p_cur.extend(block_log_p)
            log_target_cur.extend(block_log_target)

            if self.debug:
                print(f"[MCMC] Block {block_num+1}/{num_blocks_to_generate}: generated {len(block_tokens)} tokens, total={len(tokens_cur)}")

            # Run MCMC refinement steps on current state
            for step in range(self.mcmc_steps):
                # Block-aligned index selection
                num_complete_blocks = len(tokens_cur) // self.block_size
                attempts += 1

                # Pick random block boundary (keep at least first block)
                # If restrict_to_last_n is set, only resample from last N blocks
                if self.restrict_to_last_n is not None:
                    min_block = max(1, num_complete_blocks - self.restrict_to_last_n)
                else:
                    min_block = 0

                # Check if we have a valid range
                if min_block > num_complete_blocks - 1:
                    if self.debug:
                        print(f"[MCMC]   Step {step+1}: Skipping, restrict_to_last_n={self.restrict_to_last_n} too small")
                    continue

                block_idx = random.randint(min_block, num_complete_blocks - 1)
                target_idx = block_idx * self.block_size

                # Find safe tokenization boundary near target index
                # This prevents tokenization drift that would invalidate MH ratio
                idx, prefix = self._find_safe_prefix(tokens_cur, target_idx, self.block_size)

                # Target length for proposal (same as current)
                target_len = len(tokens_cur) - idx #+ self.block_size

                # Generate new suffix
                new_suffix, tokens_prop, log_p_prop, log_target_prop, pt, ct, _ = self._sample_continuation(
                    client, prompt, prefix, target_len
                )
                total_prompt_tokens += pt
                total_completion_tokens += ct

                # Slice current suffix to match proposal length (handles variable-length proposals)
                prop_len = len(tokens_prop)
                log_p_cur_suffix = log_p_cur[idx:idx + prop_len]
                log_target_cur_suffix = log_target_cur[idx:idx + prop_len]

                # MH acceptance ratio for suffixes only
                # log A = log(π(suffix')/π(suffix)) + log(q(suffix)/q(suffix'))
                log_r = (
                    sum(log_target_prop) + sum(log_p_cur_suffix)
                    - sum(log_target_cur_suffix) - sum(log_p_prop)
                )

                # Accept with probability min(1, exp(log_r))
                accepted = np.random.rand() < np.exp(log_r)

                if self.debug:
                    status = "ACCEPT" if accepted else "REJECT"
                    print(f"[MCMC]   Step {step+1}: block_idx={block_idx}, idx={idx}, log_r={log_r:.3f}, {status}")

                if accepted:
                    acceptances += 1
                    # Update current state with new suffix
                    tokens_cur = tokens_cur[:idx] + tokens_prop
                    log_p_cur = log_p_cur[:idx] + log_p_prop
                    log_target_cur = log_target_cur[:idx] + log_target_prop

        # Reconstruct text from final tokens
        current_text = "".join(tokens_cur)

        # Store acceptance ratio for diagnostics
        self._last_acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0

        if self.debug:
            print(f"[MCMC] Final: {len(tokens_cur)} tokens, acceptance={self._last_acceptance_ratio:.1%}")
            print(f"[MCMC] Final text: {current_text[:200]}..." if len(current_text) > 200 else f"[MCMC] Final text: {current_text}")

        return current_text, total_prompt_tokens, total_completion_tokens

    def get_acceptance_ratio(self) -> float:
        """Return the acceptance ratio from the last generate() call."""
        return getattr(self, '_last_acceptance_ratio', 0.0)


@dataclass
class Proposal:
    """A proposal for parallel MCMC."""
    tokens: List[str]
    log_p: List[float]
    log_target: List[float]
    prompt_tokens: int
    completion_tokens: int

class ParallelMCMCSampling(SamplingStrategy):
    """
    Parallel MCMC with multiple proposals per step.

    Implements the MH acceptance ratio from "Reasoning with Sampling" (Eq. 9):

    A(x, x') = min{1, [p(x')^α · q(x|x')] / [p(x)^α · q(x'|x)]}

    With independent proposal q(x) = p(x):

    log R = α·log P(x') + log P(x) - α·log P(x) - log P(x')
          = log_target' + log_p - log_target - log_p'

    Uses Calderhead's parallel structure: generate N proposals, build transition
    matrix, sample next state. This enables parallel generation while maintaining
    correct MH dynamics.
    """

    def __init__(
        self,
        alpha: float = 1.67,
        mcmc_steps: int = 5,
        top_logprobs: int = 5,
        proposal_temperature: float = 1.0,  # Usually want higher temp for proposals to explore
        block_size: int = 16,  # Smaller blocks work better for MCMC
        debug: bool = False,
        num_proposals: int = 4,
        max_concurrent: int = 100,
        timeout: float = 60.0,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        supports_n_param: bool = True,  # Whether API supports n parameter for batching
        seed: int = None,  # Random seed for API reproducibility
        use_length_penalty: bool = False,  # Whether to apply length normalization
        length_penalty: float = 0.6,  # Length penalty exponent (like beam search)
    ):
        name = f"ParallelMCMC(α={alpha},steps={mcmc_steps},B={block_size},N={num_proposals})"
        super().__init__(name, seed=seed)
        self.alpha = alpha
        self.mcmc_steps = mcmc_steps
        self.top_logprobs = top_logprobs
        self.proposal_temperature = proposal_temperature
        self.block_size = block_size
        self.debug = debug
        self.num_proposals = num_proposals
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.supports_n_param = supports_n_param
        self.use_length_penalty = use_length_penalty
        self.length_penalty = length_penalty

    def _extract_logprobs_completion(self, choice: Any) -> Tuple[List[str], List[float]]:
        """
        Extract tokens and logprobs from completions API choice.
        """
        try:
            if not choice.logprobs:
                return [], []

            tokens = choice.logprobs.tokens
            log_p_raw = choice.logprobs.token_logprobs

            # Filter out None values (first token often has None logprob)
            valid = [(t, lp) for t, lp in zip(tokens, log_p_raw) if lp is not None]
            if not valid:
                return [], []

            tokens, log_ps = zip(*valid)
            return list(tokens), list(log_ps)
        except Exception as e:
            if self.debug:
                print(f"Error extracting logprobs: {e}")
            return [], []

    async def _call_api(
        self,
        client: Any,
        prompt: str,
        prefix: str,
        max_tokens: int
    ) -> Any:
        """API call using completions API for TRUE continuation."""
        full_prompt = self._apply_chat_template(prompt, prefix)
        return await client.completions.create(
            model=self.model,
            prompt=full_prompt,
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
            seed=self.seed,
        )

    async def _call_api_multiple(
        self,
        client: Any,
        prompt: str,
        prefix: str,
        max_tokens: int,
        n: int
    ) -> Any:
        """API call with n parameter for multiple completions using completions API."""
        full_prompt = self._apply_chat_template(prompt, prefix)
        return await client.completions.create(
            model=self.model,
            prompt=full_prompt,
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
            n=n,
            seed=self.seed,
        )

    async def _generate_parallel_proposals(
        self,
        client: Any,
        prompt: str,
        prefix: str,
        target_len: int,
        current_proposal: Proposal,
    ) -> Tuple[List[Proposal], int, int]:
        """
        Generate N-1 new proposals in parallel + include current state.

        When supports_n_param=True (vLLM): Use single call with n parameter for efficient GPU batching.
        When supports_n_param=False: Make separate parallel API calls.

        Uses completions API for TRUE continuation.
        """
        proposals = [current_proposal]
        total_pt = 0
        total_ct = 0

        if self.supports_n_param:
            # vLLM path: Use n parameter for efficient batched inference
            try:
                resp = await self._call_api_multiple(client, prompt, prefix, target_len, self.num_proposals - 1)

                for choice in resp.choices:
                    tokens, log_p = self._extract_logprobs_completion(choice)
                    if not tokens:
                        continue

                    log_target = [self.alpha * lp for lp in log_p]
                    proposals.append(Proposal(
                        tokens=tokens,
                        log_p=log_p,
                        log_target=log_target,
                        prompt_tokens=resp.usage.prompt_tokens // len(resp.choices),
                        completion_tokens=resp.usage.completion_tokens // len(resp.choices),
                    ))

                total_pt = resp.usage.prompt_tokens
                total_ct = resp.usage.completion_tokens

            except Exception as e:
                if self.debug:
                    print(f"API Error with n param: {e}")
        else:
            # Fallback path: Make N-1 separate parallel API calls
            tasks = []
            for _ in range(self.num_proposals - 1):
                tasks.append(self._call_api(client, prompt, prefix, target_len))

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for resp in responses:
                if isinstance(resp, Exception):
                    if self.debug:
                        print(f"API Error: {resp}")
                    continue

                tokens, log_p = self._extract_logprobs_completion(resp.choices[0])
                if not tokens:
                    continue

                log_target = [self.alpha * lp for lp in log_p]
                proposals.append(Proposal(
                    tokens=tokens,
                    log_p=log_p,
                    log_target=log_target,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                ))
                total_pt += resp.usage.prompt_tokens
                total_ct += resp.usage.completion_tokens

        return proposals, total_pt, total_ct

    def _compute_transition_matrix(self, proposals: List[Proposal]) -> np.ndarray:
        """
        Computes transition matrix for parallel MCMC with MH acceptance.

        Following "Reasoning with Sampling" (Eq. 9):

        A(x, x') = min{1, [p(x')^α · q(x|x')] / [p(x)^α · q(x'|x)]}

        With independent proposal q(x) = p(x), in log space:

        log R(i,j) = [α·log P(j) + log P(i)] - [α·log P(i) + log P(j)]
                   = log_target_j + log_p_i - log_target_i - log_p_j

        This matches the serial MCMC's acceptance ratio formula.

        Transition probabilities (Calderhead structure):
        A(i,j) = (1/N) * min(1, R(i,j))  for i ≠ j
        A(i,i) = 1 - Σ_{j≠i} A(i,j)
        """
        N = len(proposals)
        A = np.zeros((N, N))

        # Sum log probs for each proposal, with optional length penalty
        log_p_list = []
        log_target_list = []
        for p in proposals:
            lp = sum(p.log_p)
            lt = sum(p.log_target)
            if self.use_length_penalty and len(p.tokens) > 0:
                # Normalize by length^penalty (like beam search)
                length_norm = len(p.tokens) ** self.length_penalty
                lp = lp / length_norm
                lt = lt / length_norm
            log_p_list.append(lp)
            log_target_list.append(lt)

        # Compute transition matrix using MH ratio (matching serial MCMC)
        for i in range(N):
            for j in range(N):
                if i != j:
                    # log R(i→j) = log_target_j + log_p_i - log_target_i - log_p_j
                    log_R = (log_target_list[j] + log_p_list[i]
                             - log_target_list[i] - log_p_list[j])
                    log_R = np.clip(log_R, -50, 50)
                    A[i, j] = (1.0 / N) * min(1.0, np.exp(log_R))

            A[i, i] = 1.0 - np.sum(A[i, :])

        return np.maximum(A, 0.0)

    async def _generate_async(self, client: Any, prompt: str, max_tokens: int) -> Tuple[str, int, int]:
        """
        Generate completion using parallel MCMC with block-wise generation.

        Algorithm (matching serial MCMCSampling structure):
        1. Generate tokens block-by-block (B tokens per block)
        2. After each block, run MCMC refinement steps with N parallel proposals
        3. MCMC uses block-aligned index selection
        4. Uses Calderhead transition matrix for proposal selection

        Uses completions API for TRUE continuation.
        """
        tokens_cur: List[str] = []
        log_p_cur: List[float] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        attempts = 0
        acceptances = 0

        # Calculate number of blocks to generate
        num_blocks_to_generate = max_tokens // self.block_size
        if num_blocks_to_generate < 1:
            num_blocks_to_generate = 1

        if self.debug:
            print(f"[ParallelMCMC] Block-wise generation: {num_blocks_to_generate} blocks of {self.block_size} tokens")

        # Generate block by block
        for block_num in range(num_blocks_to_generate):
            # Generate next block - validate tokenization boundary
            if tokens_cur:
                _, prefix = self._find_safe_prefix(tokens_cur, len(tokens_cur), self.block_size)
            else:
                prefix = ""

            block_resp = await self._call_api(client, prompt, prefix, self.block_size)
            block_tokens, block_log_p = self._extract_logprobs_completion(block_resp.choices[0])
            total_prompt_tokens += block_resp.usage.prompt_tokens
            total_completion_tokens += block_resp.usage.completion_tokens

            # Extend current state with new block
            tokens_cur.extend(block_tokens)
            log_p_cur.extend(block_log_p)

            if self.debug:
                print(f"[ParallelMCMC] Block {block_num+1}/{num_blocks_to_generate}: "
                      f"generated {len(block_tokens)} tokens, total={len(tokens_cur)}")

            # Run MCMC refinement steps on current state
            for step in range(self.mcmc_steps):
                num_complete_blocks = len(tokens_cur) // self.block_size

                # Skip if no complete blocks to resample from
                if num_complete_blocks < 1:
                    if self.debug:
                        print(f"[ParallelMCMC] Step {step+1}: not skipping but warning")
                    # continue

                # Block-aligned index selection
                block_idx = random.randint(0, num_complete_blocks - 1) if num_complete_blocks > 0 else 0
                target_idx = block_idx * self.block_size

                # Find safe tokenization boundary near target index
                # This prevents tokenization drift that would invalidate MH ratio
                pivot_idx, prefix = self._find_safe_prefix(tokens_cur, target_idx, self.block_size)

                suffix_tokens = tokens_cur[pivot_idx:]
                suffix_log_p = log_p_cur[pivot_idx:]

                # Target length: same as current suffix (strict Calderhead)
                target_gen_len = len(suffix_tokens)

                current_proposal = Proposal(
                    tokens=suffix_tokens,
                    log_p=suffix_log_p,
                    log_target=[self.alpha * x for x in suffix_log_p],
                    prompt_tokens=0, completion_tokens=0
                )

                # Generate N-1 parallel proposals
                proposals, pt, ct = await self._generate_parallel_proposals(
                    client, prompt, prefix, target_gen_len, current_proposal
                )
                total_prompt_tokens += pt
                total_completion_tokens += ct

                if len(proposals) < 2:
                    continue

                # Calderhead transition
                A = self._compute_transition_matrix(proposals)
                attempts += 1
                next_idx = np.random.choice(len(proposals), p=A[0])  # 0 is current

                if self.debug:
                    log_targets = [sum(p.log_target) for p in proposals]
                    status = f"ACCEPT(proposal {next_idx})" if next_idx != 0 else "STAY"
                    print(f"[ParallelMCMC]   Step {step+1}: block_idx={block_idx}, pivot_idx={pivot_idx}, "
                          f"log_targets={[f'{lt:.1f}' for lt in log_targets]}, {status}")

                if next_idx != 0:
                    acceptances += 1
                    selected = proposals[next_idx]
                    # Update state with selected proposal
                    tokens_cur = tokens_cur[:pivot_idx] + selected.tokens
                    log_p_cur = log_p_cur[:pivot_idx] + selected.log_p

        # Store acceptance ratio
        self._last_acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0

        if self.debug:
            print(f"[ParallelMCMC] Final: {len(tokens_cur)} tokens, acceptance={self._last_acceptance_ratio:.1%}")

        print("AAAAAAAAAAAAA", "".join(tokens_cur))
        return "".join(tokens_cur), total_prompt_tokens, total_completion_tokens

    async def _run_with_async_client(self, prompt: str, max_tokens: int) -> Tuple[str, int, int]:
        """Helper to run async generation with proper AsyncOpenAI client."""
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as async_client:
            return await self._generate_async(async_client, prompt, max_tokens)

    def generate(self, client: Any, prompt: str, max_tokens: int = 512) -> Tuple[str, int, int]:
        """Sync entry point. Creates its own AsyncOpenAI client internally."""
        # Extract api_key and base_url from passed client if not already set
        if self.api_key is None:
            self.api_key = client.api_key
        if self.base_url == "https://api.openai.com/v1":
            self.base_url = str(client.base_url)
        return asyncio.run(self._run_with_async_client(prompt, max_tokens))

@dataclass
class Beam:
    """Represents a single hypothesis in beam search."""
    text: str = ""
    tokens: list[str] = field(default_factory=list)
    log_p: list[float] = field(default_factory=list)
    log_target: list[float] = field(default_factory=list)
    finished: bool = False

    def __len__(self) -> int:
        """Return number of tokens."""
        return len(self.tokens)

    def extend(self, text: str, tokens: list[str], log_p: list[float],
               log_target: list[float], finished: bool) -> "Beam":
        """Create a new beam by appending a continuation."""
        return Beam(
            text=self.text + text,
            tokens=self.tokens + tokens,
            log_p=self.log_p + log_p,
            log_target=self.log_target + log_target,
            finished=finished,
        )

    def score(self, use_length_penalty: bool = False,
              length_penalty: float = 0.6) -> float:
        """Calculate beam score with optional length normalization."""
        if len(self) == 0:
            return float('-inf')
        cumulative = sum(self.log_target)
        if use_length_penalty:
            return cumulative / (len(self) ** length_penalty)
        return cumulative


class BeamSearchSampling(SamplingStrategy):
    """
    Beam search with power sampling via API.

    Maintains beam_width parallel hypotheses, scores using p^α logprobs,
    and uses length normalization to prevent short-sequence bias.

    Unlike MCMC which uses accept/reject, beam search deterministically
    keeps the top-k hypotheses at each step.
    """

    def __init__(
        self,
        alpha: float = 4.0,
        beam_width: int = 2,
        n_per_beam: int = 2,  # Generate n continuations per beam
        tokens_per_step: int = 192,  # Generate this many tokens per expansion
        use_length_penalty: bool = True,  # Whether to apply length normalization
        length_penalty: float = 0.6,
        proposal_temperature: float = 1.0,
        top_logprobs: int = 5,
        debug: bool = False,
        supports_n_param: bool = True,  # Whether API supports n parameter for batching
        max_concurrent: int = 100,  # Max concurrent API requests
        timeout: float = 300.0,  # Timeout in seconds (longer for local servers)
        seed: int = None,  # Random seed for API reproducibility
    ):
        name = f"BeamSearch(α={alpha},width={beam_width},n={n_per_beam},tps={tokens_per_step})"
        super().__init__(name, seed=seed)
        self.alpha = alpha
        self.beam_width = beam_width
        self.n_per_beam = n_per_beam
        self.tokens_per_step = tokens_per_step
        self.use_length_penalty = use_length_penalty
        self.length_penalty = length_penalty
        self.proposal_temperature = proposal_temperature
        self.top_logprobs = top_logprobs
        self.debug = debug
        self.supports_n_param = supports_n_param
        self.max_concurrent = max_concurrent
        self.timeout = timeout

    def _extract_logprobs_completion(self, choice) -> tuple[list[str], list[float], list[float]]:
        """Extract tokens and logprobs from completions API choice."""
        if not choice.logprobs:
            return [], [], []

        tokens = choice.logprobs.tokens
        log_p_raw = choice.logprobs.token_logprobs

        # Filter out None values (first token often has None logprob)
        valid = [(t, lp) for t, lp in zip(tokens, log_p_raw) if lp is not None]
        if not valid:
            return [], [], []

        tokens, log_p = zip(*valid)
        tokens, log_p = list(tokens), list(log_p)
        log_target = [self.alpha * lp for lp in log_p]

        return tokens, log_p, log_target

    def _sample_full(self, client: OpenAI, prompt: str, max_tokens: int):
        """Generate a full sample from base model using completions API."""
        full_prompt = self._apply_chat_template(prompt)
        response = client.completions.create(
            model=client.default_model,
            prompt=full_prompt,
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
            seed=self.seed,
        )

        text = response.choices[0].text
        tokens, log_p, log_target = self._extract_logprobs_completion(response.choices[0])
        finished_naturally = response.choices[0].finish_reason == "stop"

        return (
            text, tokens, log_p, log_target,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            finished_naturally
        )

    def _sample_continuation(self, client: OpenAI, prompt: str, prefix: str, max_tokens: int):
        """Generate a TRUE continuation from a prefix using completions API."""
        full_prompt = self._apply_chat_template(prompt, prefix)
        response = client.completions.create(
            model=client.default_model,
            prompt=full_prompt,
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
            seed=self.seed,
        )

        continuation = response.choices[0].text
        tokens, log_p, log_target = self._extract_logprobs_completion(response.choices[0])
        finished_naturally = response.choices[0].finish_reason == "stop"

        return (
            continuation, tokens, log_p, log_target,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            finished_naturally
        )

    def _sample_continuation_multiple(self, client: OpenAI, prompt: str, prefix: str, max_tokens: int, n: int):
        """Generate n TRUE continuations from a prefix using completions API with n parameter."""
        full_prompt = self._apply_chat_template(prompt, prefix)
        response = client.completions.create(
            model=client.default_model,
            prompt=full_prompt,
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
            n=n,  # Generate n different samples
            seed=self.seed,
        )

        # Extract all n continuations
        results = []
        for choice in response.choices:
            continuation = choice.text
            tokens, log_p, log_target = self._extract_logprobs_completion(choice)
            finished_naturally = choice.finish_reason == "stop"
            results.append((continuation, tokens, log_p, log_target, finished_naturally))

        # Token usage is for ALL n samples combined
        return results, response.usage.prompt_tokens, response.usage.completion_tokens

    async def _sample_single_continuation_async(self, client: AsyncOpenAI, prompt: str, prefix: str, max_tokens: int):
        """Async: Generate a TRUE continuation from a prefix using completions API."""
        full_prompt = self._apply_chat_template(prompt, prefix)
        response = await client.completions.create(
            model=client.default_model,
            prompt=full_prompt,
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
            seed=self.seed,
        )

        choice = response.choices[0]
        continuation = choice.text
        tokens, log_p, log_target = self._extract_logprobs_completion(choice)
        finished_naturally = choice.finish_reason == "stop"

        return (
            continuation, tokens, log_p, log_target, finished_naturally,
            response.usage.prompt_tokens, response.usage.completion_tokens
        )

    async def _sample_multiple_continuations_async(self, client: AsyncOpenAI, prompt: str, prefix: str, max_tokens: int, n: int):
        """Async: Generate n TRUE continuations using completions API with n parameter."""
        full_prompt = self._apply_chat_template(prompt, prefix)
        response = await client.completions.create(
            model=client.default_model,
            prompt=full_prompt,
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
            n=n,
            seed=self.seed,
        )

        results = []
        for choice in response.choices:
            continuation = choice.text
            tokens, log_p, log_target = self._extract_logprobs_completion(choice)
            finished_naturally = choice.finish_reason == "stop"
            results.append((continuation, tokens, log_p, log_target, finished_naturally))

        return results, response.usage.prompt_tokens, response.usage.completion_tokens

    async def _expand_beams_parallel(self, client: AsyncOpenAI, active_beams: list[Beam], prompt: str, block_num: int) -> tuple[list[Beam], int, int]:
        """
        Parallelize beam expansion.

        When supports_n_param=True (vLLM): Use n parameter for efficient GPU batching.
        When supports_n_param=False: Make separate parallel API calls.
        """
        candidate_beams: list[Beam] = []
        total_pt = 0
        total_ct = 0

        if self.supports_n_param:
            # vLLM path: Use n parameter for efficient batched inference
            # Make beam_width parallel calls, each with n=n_per_beam
            tasks = []
            beam_refs: list[Beam] = []  # Track which beam each task belongs to

            for beam in active_beams:
                prefix = "" if (block_num == 0 and not beam.text) else beam.text
                task = self._sample_multiple_continuations_async(
                    client, prompt, prefix, self.tokens_per_step, self.n_per_beam
                )
                tasks.append(task)
                beam_refs.append(beam)

            # Run all beam expansions in parallel
            results = await asyncio.gather(*tasks)

            for beam, (continuations, pt, ct) in zip(beam_refs, results):
                total_pt += pt
                total_ct += ct

                for text, tokens, log_p, log_target, finished in continuations:
                    if block_num == 0:
                        candidate_beams.append(Beam(text, tokens, log_p, log_target, finished))
                    else:
                        candidate_beams.append(beam.extend(text, tokens, log_p, log_target, finished))
        else:
            # Fallback path: Make beam_width * n_per_beam separate parallel API calls
            all_tasks = []

            for beam in active_beams:
                prefix = "" if (block_num == 0 and not beam.text) else beam.text

                # Make n_per_beam separate calls for this beam
                for _ in range(self.n_per_beam):
                    task = self._sample_single_continuation_async(
                        client, prompt, prefix, self.tokens_per_step
                    )
                    all_tasks.append((task, beam))

            # Run ALL API calls in parallel
            results = await asyncio.gather(*[task for task, _ in all_tasks])

            for i, result in enumerate(results):
                text, tokens, log_p, log_target, finished, pt, ct = result
                total_pt += pt
                total_ct += ct
                beam = all_tasks[i][1]

                if block_num == 0:
                    candidate_beams.append(Beam(text, tokens, log_p, log_target, finished))
                else:
                    tmp_beam = beam.extend(text, tokens, log_p, log_target, finished)
                    print(f'tmp_beam: {tmp_beam.text}')
                    candidate_beams.append(tmp_beam)

        return candidate_beams, total_pt, total_ct

    async def _generate_async(self, client: AsyncOpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        """
        Async version of generate() with parallel beam expansion.

        Algorithm:
        1. Start with beam_width INDEPENDENT samples (not 1 empty beam)
        2. For each beam, generate n_per_beam continuations IN PARALLEL (beam branching)
        3. Score all beam_width × n_per_beam candidates using p^α
        4. Keep top beam_width beams by score
        5. Repeat until max_tokens or all beams finish

        With beam_width=10, n_per_beam=3: generates 30 candidates per iteration in parallel.
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Will be initialized below with beam_width independent samples
        completed_beams: list[tuple[float, Beam]] = []  # (score, beam) pairs

        num_expansions = 0
        num_blocks = max_tokens // self.tokens_per_step
        if num_blocks < 1:
            num_blocks = 1

        if self.debug:
            print(f"[BeamSearch] TRUE beam search (ASYNC): beam_width={self.beam_width}, n_per_beam={self.n_per_beam}")
            print(f"[BeamSearch] Generating {num_blocks} blocks of {self.tokens_per_step} tokens")
            print(f"[BeamSearch] α={self.alpha}, length_penalty={self.length_penalty}")

        # INITIALIZATION: Generate beam_width INDEPENDENT samples to start with diversity
        if self.debug:
            print(f"[BeamSearch] Initializing with {self.beam_width} independent samples...")

        init_results, init_pt, init_ct = await self._sample_multiple_continuations_async(
            client, prompt, "", self.tokens_per_step, self.beam_width
        )
        total_prompt_tokens += init_pt
        total_completion_tokens += init_ct

        # Convert to Beam objects and score
        scored_init: list[tuple[float, Beam]] = []
        for text, tokens, log_p, log_target, finished in init_results:
            beam = Beam(text, tokens, log_p, log_target, finished)
            score = beam.score(self.use_length_penalty, self.length_penalty)
            scored_init.append((score, beam))

        # Sort by score and separate completed vs active
        scored_init.sort(key=lambda x: x[0], reverse=True)
        completed_beams = [(s, b) for s, b in scored_init if b.finished]
        active_beams: list[Beam] = [b for s, b in scored_init if not b.finished][:self.beam_width]

        if self.debug:
            print(f"[BeamSearch] Initialized {len(active_beams)} active beams, {len(completed_beams)} already completed")
            if scored_init:
                print(f"[BeamSearch] Initial best score: {scored_init[0][0]:.4f}")
            print(f"[BeamSearch] Each iteration: {len(active_beams)} beams × {self.n_per_beam} samples = candidates (PARALLEL)")

        # Start from block 1 since block 0 was the initialization
        for block_num in range(1, num_blocks):
            if not active_beams:
                break

            # PARALLEL EXPANSION PHASE: Generate n_per_beam continuations for ALL beams at once
            candidate_beams, pt, ct = await self._expand_beams_parallel(
                client, active_beams, prompt, block_num
            )
            total_prompt_tokens += pt
            total_completion_tokens += ct

            num_expansions += 1

            if self.debug:
                print(f"[BeamSearch] Block {block_num+1}/{num_blocks}: Generated {len(candidate_beams)} candidates (parallel)")

            # Score all candidates
            scored_beams: list[tuple[float, Beam]] = [
                (beam.score(self.use_length_penalty, self.length_penalty), beam)
                for beam in candidate_beams
            ]

            # Sort by score (descending)
            scored_beams.sort(key=lambda x: x[0], reverse=True)

            # Separate completed and active
            new_completed = [(s, b) for s, b in scored_beams if b.finished]
            new_active = [b for s, b in scored_beams if not b.finished]

            # Keep top beams
            completed_beams.extend(new_completed[:self.beam_width])
            active_beams = new_active[:self.beam_width]

            if self.debug:
                print(f"[BeamSearch] Block {block_num+1}/{num_blocks}: "
                      f"{len(active_beams)} active, {len(completed_beams)} completed")
                if scored_beams:
                    best_score = scored_beams[0][0]
                    print(f"[BeamSearch]   Best score: {best_score:.4f}")

            # Stop if we have enough completed beams
            if len(completed_beams) >= self.beam_width:
                if self.debug:
                    print(f"[BeamSearch] Stopping: {len(completed_beams)} beams completed")
                break

        # Select best beam from all (completed + active)
        all_beams: list[tuple[float, Beam]] = completed_beams + [
            (beam.score(self.use_length_penalty, self.length_penalty), beam)
            for beam in active_beams
        ]

        if not all_beams:
            # Fallback: return empty
            return "", total_prompt_tokens, total_completion_tokens

        # Sort by score and take best
        all_beams.sort(key=lambda x: x[0], reverse=True)
        print(f'{[score for score, x in all_beams]}')
        best_score, best_beam = all_beams[0]

        # Store metadata for diagnostics
        self._last_num_expansions = num_expansions
        self._last_best_score = best_score
        self._last_num_completed = len(completed_beams)

        if self.debug:
            print(f"[BeamSearch] Final: {len(best_beam)} tokens, score={best_score:.4f}")
            print(f"[BeamSearch] Text: {best_beam.text[:200]}..." if len(best_beam.text) > 200 else f"[BeamSearch] Text: {best_beam.text}")

        print(f'best beam text: {best_beam.text}')
        return best_beam.text, total_prompt_tokens, total_completion_tokens

    async def _run_with_client(self, api_key: str, base_url: str, model: str, prompt: str, max_tokens: int):
        """Helper to run async generation with proper client lifecycle."""
        # Configure httpx client with larger connection pool and longer timeout for local servers
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self.max_concurrent + 10,
                max_keepalive_connections=self.max_concurrent,
            ),
            timeout=httpx.Timeout(self.timeout, connect=30.0),
        )
        async with AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
            http_client=http_client,
        ) as async_client:
            async_client.default_model = model
            return await self._generate_async(async_client, prompt, max_tokens)

    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        """
        Generate completion using TRUE beam search with power sampling (with async parallelization).

        This method wraps the async implementation and creates an AsyncOpenAI client automatically.

        Algorithm:
        1. Start with empty beam
        2. For each beam, generate n_per_beam continuations IN PARALLEL (beam branching)
        3. Score all beam_width × n_per_beam candidates using p^α
        4. Keep top beam_width beams by score
        5. Repeat until max_tokens or all beams finish

        With beam_width=2, n_per_beam=2: generates 4 candidates per iteration in parallel.
        """
        # Run async version with proper client lifecycle management
        return asyncio.run(
            self._run_with_client(
                api_key=client.api_key,
                base_url=str(client.base_url) if client.base_url else "https://api.openai.com/v1",
                model=client.default_model,
                prompt=prompt,
                max_tokens=max_tokens
            )
        )

    def get_num_expansions(self) -> int:
        """Return the number of expansions from the last generate() call."""
        return getattr(self, '_last_num_expansions', 0)

    def get_best_score(self) -> float:
        """Return the best beam score from the last generate() call."""
        return getattr(self, '_last_best_score', 0.0)


class TemperatureSampling(SamplingStrategy):
    """Standard temperature sampling using completions API."""

    def __init__(self, temperature: float = 0.8, seed: int = None):
        super().__init__(f"Temperature(T={temperature})", seed=seed)
        self.temperature = temperature

    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        full_prompt = self._apply_chat_template(prompt)
        response = client.completions.create(
            model=client.default_model,
            prompt=full_prompt,
            temperature=self.temperature,
            max_tokens=max_tokens,
            seed=self.seed,
        )
        return (
            response.choices[0].text,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )


class Benchmark(ABC):
    """Abstract base class for benchmarks."""
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of the benchmark."""
        pass
    
    @abstractmethod
    def load_dataset(self):
        """Load the benchmark dataset."""
        pass
    
    @abstractmethod
    def get_problem(self, index: int) -> Dict:
        """Get a problem by index."""
        pass
    
    @abstractmethod
    def get_num_problems(self) -> int:
        """Return total number of problems in the dataset."""
        pass
    
    @abstractmethod
    def format_prompt(self, problem: Dict) -> str:
        """Format a problem into a prompt for the LLM."""
        pass
    
    @abstractmethod
    def extract_completion(self, response: str, problem: Dict) -> str:
        """Extract the completion from LLM response."""
        pass
    
    @abstractmethod
    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.
        Returns: (passed, result_message)
        """
        pass
    
    @abstractmethod
    def format_prediction(self, problem: Dict, completion: str) -> Dict:
        """
        Format a prediction for official evaluation tools.
        Returns: Dictionary in the format expected by the benchmark's evaluator.
        """
        pass


class HumanEvalBenchmark(Benchmark):
    """HumanEval benchmark implementation."""
    
    def __init__(self):
        self.dataset = None
    
    def name(self) -> str:
        return "HumanEval"
    
    def load_dataset(self):
        if self.dataset is None:
            self.dataset = load_dataset("openai/openai_humaneval", split="test")
    
    def get_problem(self, index: int) -> Dict:
        return self.dataset[index]
    
    def get_num_problems(self) -> int:
        return len(self.dataset)
    
    def format_prompt(self, problem: Dict) -> str:
        """For HumanEval, the prompt is already in the problem."""
        return problem["prompt"]
    
    def extract_completion(self, response: str, problem: Dict) -> str:
        """Extract code completion from LLM response."""
        return extract_code_completion(response, problem["entry_point"])
    
    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """
        DEPRECATED: Use official evaluation instead.
        This method is not reliable - use format_prediction() and official evaluators.
        """
        # Return None to indicate evaluation should be done externally
        return False, "use_official_evaluator"
    
    def format_prediction(self, problem: Dict, completion: str) -> Dict:
        """Format prediction for HumanEval official evaluator."""
        return {
            "task_id": problem["task_id"],
            "completion": completion
        }


def extract_code_completion(response: str, entry_point: str) -> str:
    """Extract code completion from LLM response."""
    # Try to find code blocks
    code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    code_blocks = re.findall(r'```\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # If no code blocks, look for function definition
    lines = response.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if f'def {entry_point}' in line:
            in_function = True
        if in_function:
            code_lines.append(line)
            # Stop at next function or class definition
            if line.strip().startswith('def ') and f'def {entry_point}' not in line:
                break
            if line.strip().startswith('class '):
                break
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # Fallback: return the whole response
    return response.strip()


def check_code_execution(problem: Dict, completion: str, timeout: float = 3.0) -> tuple[bool, str]:
    """
    Simple code execution checker.
    Returns: (passed, result_message)
    """
    check_program = (
        problem["prompt"]
        + "\n"
        + completion
        + "\n"
        + problem["test"]
        + "\n"
        + f"check({problem['entry_point']})"
    )
    
    try:
        exec_globals = {}
        exec(check_program, exec_globals)
        return True, "passed"
    except Exception as e:
        return False, f"failed: {str(e)[:100]}"


class BenchmarkRunner:
    """Runner for comparing different sampling strategies on any benchmark."""
    
    def __init__(
        self,
        benchmark: Benchmark,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        output_dir: str = "predictions",
        prompt_prefix: str = "",
        prompt_suffix: str = "",
        suffix_overrides: Optional[Dict[str, str]] = None  # Strategy name -> suffix override
    ):
        self.benchmark = benchmark
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.client.default_model = model_name
        self.results: List[SamplingResult] = []
        self.output_dir = output_dir
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.suffix_overrides = suffix_overrides or {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Auto-load tokenizer for MCMC/BeamSearch strategies
        SamplingStrategy.set_tokenizer_from_model(model_name)
    
    def run_single_problem(
        self,
        problem: Dict,
        strategy: SamplingStrategy,
        max_tokens: int = 512
    ) -> SamplingResult:
        """Run a single problem with a given sampling strategy."""
        # Get task ID (benchmark-specific)
        task_id = problem.get("task_id") or problem.get("id") or str(problem)

        # Format prompt using benchmark
        prompt = self.benchmark.format_prompt(problem)

        # Apply custom prefix/suffix if provided
        # Check for strategy-specific suffix override
        suffix = self.suffix_overrides.get(strategy.name, self.prompt_suffix)
        if self.prompt_prefix:
            prompt = self.prompt_prefix + prompt
        if suffix:
            prompt = prompt + suffix
        
        # Generate completion
        start_time = time.time()
        completion, prompt_tokens, completion_tokens = strategy.generate(
            self.client, prompt, max_tokens
        )
        elapsed_time = time.time() - start_time
        
        # Extract completion using benchmark
        extracted_completion = self.benchmark.extract_completion(completion, problem)
        
        # Check correctness using benchmark
        passed, result_msg = self.benchmark.check_correctness(problem, extracted_completion)

        return SamplingResult(
            task_id=task_id,
            completion=extracted_completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            time_seconds=elapsed_time,
            passed=passed,
            metadata={"result_message": result_msg}
        )
    
    def run_benchmark(
        self,
        strategies: List[SamplingStrategy],
        num_problems: int = 10,
        max_tokens: int = 512,
        run_id: str = None
    ) -> Dict[str, BenchmarkMetrics]:
        """
        Run benchmark for multiple strategies.
        Generates prediction files for official evaluation.
        Returns: Dict mapping strategy name to metrics.
        """
        # Load benchmark dataset
        print(f"Loading {self.benchmark.name()} dataset...")
        self.benchmark.load_dataset()
        
        results_by_strategy: Dict[str, List[SamplingResult]] = {s.name: [] for s in strategies}
        predictions_by_strategy: Dict[str, List[Dict]] = {s.name: [] for s in strategies}
        
        print(f"\nRunning benchmark on {num_problems} {self.benchmark.name()} problems...")
        print(f"Model: {self.model_name}")
        print(f"Strategies: {[s.name for s in strategies]}\n")
        
        num_problems = min(num_problems, self.benchmark.get_num_problems())
        
        for i in range(num_problems):
            problem = self.benchmark.get_problem(i)
            task_id = problem.get("task_id") or problem.get("instance_id") or problem.get("id") or f"Problem {i+1}"
            print(f"\nProblem {i+1}/{num_problems}: {task_id}")
            
            for strategy in strategies:
                print(f"  Testing {strategy.name}...", end=" ")
                try:
                    result = self.run_single_problem(problem, strategy, max_tokens)
                    results_by_strategy[strategy.name].append(result)
                    
                    # Format prediction for official evaluator
                    prediction = self.benchmark.format_prediction(problem, result.completion)
                    predictions_by_strategy[strategy.name].append(prediction)
                    
                    status = "✓ PASS" if result.passed else "✗ FAIL"
                    print(f"{status} ({result.time_seconds:.2f}s, {result.total_tokens} tokens)")
                    if not result.passed and result.metadata:
                        print(f"    {result.metadata.get('result_message', '')}")
                except Exception as e:
                    print(f"✗ ERROR: {str(e)[:50]}")
        
        # Save prediction files
        for strategy_name, predictions in predictions_by_strategy.items():
            if predictions:
                self.save_predictions(predictions, strategy_name, run_id)
        
        # Aggregate metrics
        metrics = {}
        for strategy_name, results in results_by_strategy.items():
            if not results:
                continue

            # Calculate pass rate from results
            num_passed = sum(1 for r in results if r.passed)
            pass_rate = (num_passed / len(results)) * 100.0 if results else 0.0

            metrics[strategy_name] = BenchmarkMetrics(
                model_name=self.model_name,
                strategy_name=strategy_name,
                benchmark_name=self.benchmark.name(),
                pass_rate=pass_rate,
                avg_time=sum(r.time_seconds for r in results) / len(results),
                total_tokens=sum(r.total_tokens for r in results),
                avg_tokens_per_problem=sum(r.total_tokens for r in results) / len(results),
                num_problems=len(results)
            )
        
        return metrics
    
    def save_predictions(self, predictions: List[Dict], strategy_name: str, run_id: str = None):
        """Save predictions to file for official evaluation."""
        # Clean strategy name for filename
        safe_strategy = strategy_name.replace("(", "_").replace(")", "").replace("=", "").replace(",", "_").replace(" ", "")
        safe_model = self.model_name.replace("/", "_").replace("-", "_")
        safe_benchmark = self.benchmark.name().replace("-", "_").lower()
        
        # Create filename
        if run_id:
            filename = f"{safe_benchmark}_{safe_model}_{safe_strategy}_{run_id}.jsonl"
        else:
            filename = f"{safe_benchmark}_{safe_model}_{safe_strategy}.jsonl"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Write JSONL file
        with open(filepath, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
        
        print(f"\n📁 Saved predictions to: {filepath}")
        print(f"   Total predictions: {len(predictions)}")
        
        # Print evaluation command
        if self.benchmark.name() == "HumanEval":
            print(f"\n   To evaluate, run:")
            print(f"   evaluate_functional_correctness {filepath}")
        elif "SWE-bench" in self.benchmark.name():
            print(f"\n   To evaluate, run:")
            print(f"   python -m swebench.harness.run_evaluation \\")
            print(f"     --predictions_path {filepath} \\")
            print(f"     --swe_bench_tasks <path-to-tasks> \\")
            print(f"     --log_dir logs/")
        
        return filepath
    
    def save_results(self, filename: str):
        """Save detailed results to JSON."""
        with open(filename, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
