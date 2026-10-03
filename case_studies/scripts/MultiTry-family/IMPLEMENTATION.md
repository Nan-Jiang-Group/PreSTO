# Subtree Prefetching for Multiple-Try Metropolis Power Sampling

## 1. Overview

Subtree-prefetched MultiTryMH batches candidate suffixes for several possible future Markov chain transitions. After generation, the sampler resolves these transitions sequentially along the realized path. Each transition uses the same independent Multiple-Try Metropolis (MTM) selection and acceptance rule as the sequential implementation.

The method builds on the MTM framework of Liu, Liang, and Wong [1]. Its LLM suffix update follows the candidate weighting and acceptance equations in Guo et al. [2, Section 3.2, Eqs. (6)–(7)]. The implementation uses uniform cuts over the accumulated response and a fixed proposal distribution. Entropy-based block skipping and position selection from the full EGPS method are outside this implementation.

Ye and Lu [3] provide the precedent for combining multiple proposals with prefetching. Their public `PMP` implementation constructs a proposal cloud and resamples cloud entries using computed weights. The construction described here adapts prefetching to shared LLM prefixes while retaining a complete MTM update at every node. The comparison with PMP-MCMC is based on the authors' public code and publication metadata; the publisher's full text was unavailable for verification.

## 2. Target Distribution and Suffix Proposals

### 2.1. Fixed-horizon target

Let $a$ denote the prompt, $\mathcal{V}$ the token vocabulary, and $x=(x_1,\ldots,x_H)$ the current response of length $H$. For a base language model $p$ and sharpening exponent $\alpha>0$, the refinement target is

$$
\pi_H(x\mid a)
=\frac{p(x\mid a)^\alpha}{Z_H(a)},
\qquad
p(x\mid a)=\prod_{t=1}^{H}p(x_t\mid a,x_{<t}),
$$

where $Z_H(a)=\sum_{x\in\mathcal{V}^{H}}p(x\mid a)^\alpha$. Sharpening corresponds to $\alpha>1$. The exponent remains fixed throughout sampling, and its normalizing constant cancels from the MTM weights.

At each transition, the sampler draws a cut

$$
c\sim\operatorname{Uniform}\{0,\ldots,H-1\}.
$$

The cut specifies the number of retained response tokens. Thus $s_c=(a,x_{1:c})$ is the conditioning prefix and $u=x_{c+1:H}$ is the current suffix. A cut at $c=0$ regenerates the entire response. Since the cut law is uniform and the horizon is fixed, the forward and reverse cut probabilities cancel.

### 2.2. Temperature mixture

For temperature $\tau>0$ and token history $h$, define the proposal conditional

$$
q_{\tau}(v\mid h)
=\frac{\exp(\ell_v(h)/\tau)}
       {\sum_{v'\in\mathcal{V}}\exp(\ell_{v'}(h)/\tau)},
$$

where $\ell_v(h)$ is the base model's logit for token $v$. Given a fixed list of temperatures $\tau_1,\ldots,\tau_M$, each candidate independently selects one list entry uniformly and uses it for the entire suffix. For a suffix $z=(z_1,\ldots,z_{H-c})$, the resulting proposal law is

$$
q_c(z\mid s_c)
=\frac{1}{M}\sum_{m=1}^{M}
  \prod_{t=1}^{H-c}q_{\tau_m}(z_t\mid s_c,z_{<t}).
$$

The mixture therefore averages whole-suffix probabilities. Averaging token probabilities before taking their product would describe a different proposal. Each candidate selects its temperature afresh at the regeneration cut. Both proposal directions marginalize over all temperatures from that cut, including when scoring the current suffix under the reverse proposal law.

The configuration field `proposal_temperatures` specifies the list, whose default is $(0.25,0.5,1.0)$. A singleton list gives a fixed-temperature proposal. The number of candidates $K$, configured by `num_tries`, is independent of the number of temperatures $M$.

### 2.3. Cached probability evaluation

The sampler stores base-model token log probabilities and an $M\times H$ array of token log probabilities under the proposal temperatures. For each candidate it stores the corresponding suffix arrays. To score an existing suffix at a new cut, it first slices the component arrays at that cut and then computes the mixture.

For a suffix $z$, let

$$
S_{m,t}=\sum_{r=1}^{t}\log q_{\tau_m}(z_r\mid s_c,z_{<r}),
\qquad
L_t=\operatorname{LSE}(S_{1,t},\ldots,S_{M,t})-\log M,
$$

where $S_{m,0}=0$ and $\operatorname{LSE}(b_1,\ldots,b_M)=\log\sum_m\exp(b_m)$. The helper returns increments $L_t-L_{t-1}$, whose sum is $\log q_c(z\mid s_c)$. This calculation preserves the whole-suffix mixture while allowing token-aligned score storage.

## 3. Multiple-Try Metropolis Transition

At a fixed cut, draw $K$ independent candidate suffixes $y_1,\ldots,y_K$ from $q_c(\cdot\mid s_c)$. Each candidate receives the relative importance weight

$$
w_j
=\frac{p(y_j\mid s_c)^\alpha\,q_c(u\mid s_c)}
       {p(u\mid s_c)^\alpha\,q_c(y_j\mid s_c)},
\qquad
W=\sum_{j=1}^{K}w_j.
$$

The shared prefix cancels from the target ratio. Candidate selection and conditional acceptance follow Guo et al. [2, Eqs. (6)–(7)]:

$$
\Pr(J=j\mid u,y_{1:K},s_c)=\frac{w_j}{W},
\qquad
A_j=\min\left\{1,\frac{W}{1+\sum_{i\ne j}w_i}\right\}
   =\min\left\{1,\frac{W}{W-w_j+1}\right\}.
$$

With probability $A_J$, the next response is $(x_{1:c},y_J)$; otherwise it remains $x$. Because the conditional proposal is independent of the current suffix, the reverse reference set reuses the unselected candidates and includes the current suffix with relative weight one. No additional reverse candidates are generated. For $K=1$, acceptance reduces to $\min\{1,w_1\}$.

The implementation evaluates weights and acceptance in log space:

$$
\log w_j
=\alpha\bigl[\log p(y_j\mid s_c)-\log p(u\mid s_c)\bigr]
 +\log q_c(u\mid s_c)-\log q_c(y_j\mid s_c).
$$

Selection exponentiates log weights after subtracting their maximum. Acceptance evaluates the numerator and denominator with log-sum-exp, including the current state's log weight of zero in the denominator. Every visited node recomputes these relative weights against the realized current suffix, so previous accepts and rejects are reflected in subsequent decisions.

Conditioned on the current state and candidate bundle, the probabilities of the $K$ labeled acceptance outcomes and the rejection outcome are

$$
\rho_j=\frac{w_j}{W}A_j\quad(j=1,\ldots,K),
\qquad
\rho_{\mathrm{rej}}=1-\sum_{j=1}^{K}\rho_j.
$$

Candidate slots remain distinct even when their token sequences coincide. Consequently, these labeled outcome probabilities need not correspond to distinct next states.

## 4. Subtree Prefetching

### 4.1. Transition tree and persistent cuts

Each node represents one complete MTM transition and has $K+1$ children. For implementation indexing, outcomes $o=0,\ldots,K-1$ accept mathematical candidate $y_{o+1}$, and outcome $o=K$ rejects. If a node has identifier $n$, its child is

$$
\operatorname{child}(n,o)=(K+1)n+o+1.
$$

The initial node of each refinement block has identifier zero. Persistent node identifiers track the realized path and retain sampled cuts across prefetch batches within that block. Rank and display indices are local to each batch's subtree and restart at zero when the sampler reroots.

Every future node receives its own independent uniform cut, including nodes reached after rejection. A dictionary stores each cut when first drawn. If that cut is infeasible for the current prefetch batch, the sampler retains it for a later batch. Feasibility never causes a cut to be shortened or redrawn.

### 4.2. Prefix feasibility and bundle sharing

Let $x^{\mathrm{root}}$ be the response at the start of a prefetch batch. For each speculative node $v$, the planner maintains a prefix bound $b_v$: all states represented by that node agree with $x^{\mathrm{root}}$ on the first $b_v$ response tokens. The root has $b_{\mathrm{root}}=H$. For a child $v'$ of a feasible node $v$ with cut $c_v$,

$$
b_{v'}=
\begin{cases}
c_v, & \text{if the edge accepts a candidate},\\
b_v, & \text{if the edge rejects}.
\end{cases}
$$

A node's suffix bundle can be generated before its state is realized whenever

$$
c_v\le b_v.
$$

Its conditioning prefix is then known to be $(a,x^{\mathrm{root}}_{1:c_v})$. Acceptance preserves the prefix before the parent's cut, while rejection preserves the existing bound.

Feasible nodes at the same transition depth $d_v$ and cut $c_v$ share one physical bundle of $K$ candidate suffixes. They have identical conditioning prefixes and are mutually exclusive along a realized path. Bundle keys are $(d_v,c_v)$ within a batch. Including depth prevents consecutive transitions from reusing a candidate bundle. Distinct nodes retain their own current suffixes and therefore their own MTM weights even when they share proposals.

### 4.3. Scheduling and budgets

Let $B$ denote `prefetch_budget`, measured in refinement suffix requests. Each complete bundle consumes $K$ requests, giving the capacity

$$
C=\left\lfloor\frac{B}{K}\right\rfloor.
$$

The runners require $B\ge K$. Any remainder is unused. At $B=K$, a prefetch batch supplies one MTM transition. This budget excludes block extensions and additional token-scoring requests.

The request budget and the logical tree size impose different limits. With $R$ transitions remaining, the complete tree of transition nodes has

$$
N_R=\sum_{d=0}^{R-1}(K+1)^d
   =\frac{(K+1)^R-1}{K}.
$$

The planner uses $N_R$ as its node-visit bound and $C$ as its candidate-bundle capacity. Once all $C$ bundles are allocated, nodes that share an existing bundle can still be retained. Thus explicit tree exploration can require exponential CPU work even though generated refinement requests remain bounded by $B$.

The available rank policies are `accept_first`, `reject_first`, `longest_first`, `smallest_cut_diff`, `longest_path_first`, `bfs_accept_first`, and `bfs_reject_first`. Lower rank scores receive priority, with FIFO ordering for ties. The `smallest_cut_diff` policy uses the actual parent cut minus the child cut. Rank affects which bundles are scheduled; the transition probabilities remain those in Section 3.

### 4.4. Batch execution and block progression

For each batch, the engine plans feasible nodes, groups them by bundle key, and generates $K$ suffixes per key. It then commits MTM transitions sequentially along the realized path. When the next node has no prefetched bundle, the realized state becomes the root of another batch. Previously drawn cuts remain available after rerooting. Both acceptance and rejection count as completed transitions, and refinement continues until exactly `mcmc_steps` transitions have been committed for the block.

Each block first extends the response by `max_new_tokens // num_of_blocks` tokens. Refinement then operates on the entire accumulated response at its new fixed horizon. The integer-division remainder is unused. Proposals ignore end-of-sequence (EOS) tokens, and the sampler checks response EOS only after all refinement steps for the block finish. With `stop_on_eos=False`, every block runs. The returned response is always trimmed through its first EOS when an EOS token is configured. Prompt EOS tokens do not trigger response stopping.

## 5. Software Implementation

The implementation separates the transition arithmetic, prefetch planner, statistics, and backend-specific generation:

| Component | Implementation responsibility |
| --- | --- |
| [MTM helpers](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/common/multi_try.py) | Temperature-mixture scoring, candidate selection, acceptance, and labeled outcome probabilities |
| [Shared prefetch engine](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/common/prefetch_multi_try.py) | Persistent cuts, feasible bundle planning, generation requests, and realized-path updates |
| [Tree representation](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/common/prefetch_multi_try_tree.py) | Reconstruction and optional display of the prefetched tree |
| [Statistics](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/common/prefetch_multi_try_stats.py) | Backend work accounting, transition records, and JSON summaries |
| [Hugging Face adapter](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/backends/hf/samplers/subtree_prefetching_multi_try_mh.py) | Batched suffix generation and component scoring from generation logits |
| [vLLM adapter](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/power_sharpening/backends/vllm/samplers/subtree_prefetching_multi_try_mh.py) | Suffix generation and exact token-ID scoring through the existing MultiTry helpers |

The backend callback receives requests of the form `(prefix, length, temperature, seed)` and returns one `(token_ids, base_logprobs, logprobs_by_temperature)` tuple per request, together with backend work counts. Base-model scores remain separate from proposal scores. The shared engine converts returned arrays to the expected list and NumPy formats before using them in state updates. Tree reconstruction is performed only when `print_tree=True` and does not determine the sampled path.

### 5.1. Backend generation and scoring

The Hugging Face adapter batches variable-length prefixes and reuses raw generation logits to score every proposal temperature. Within each batch group, the existing helper decodes every row for the longest requested suffix and trims excess tokens. The adapter counts this overgeneration in `generated_tokens`. If padding plus decoding would exceed the context window, it separates requests by prefix length.

The vLLM adapter generates each requested suffix length directly. Generation supplies base-model scores and scores for the sampled temperature. Missing temperature components are evaluated through additional queries using exact token IDs and the patched vLLM scoring path, avoiding decode–retokenize alignment errors. Each scoring query generates one discarded token, and `scoring_batch_size` bounds the number of such requests per scoring call. The adapter uses unconstrained temperature-scaled softmax proposals so generation and proposal scoring describe the same distribution.

The [case-study launcher](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies/scripts/MultiTry-family/run_multitry_case_study.sh) configures comparisons with `cut_dist_type=uniform` and `temperature_schedule_type=const`. These settings match the shared-prefix construction above.

### 5.2. Randomness and reproducibility

The shared engine uses the supplied NumPy generator or creates `np.random.default_rng(seed)`. This stream supplies cuts, temperature choices, request seeds, candidate selection, and acceptance draws in execution order. A fixed seed and scheduling configuration reproduce these control draws. Changing the prefetch budget or rank can change stream consumption and the realized trajectory.

vLLM receives a separately drawn seed for each suffix request. The existing Hugging Face batch helper uses the global Torch random stream, which its runner also seeds, and does not support a separate generator for each row. Backend numerical differences can affect token draws. Reproducibility therefore depends on the backend and execution configuration as well as the random seed.

## 6. Work Accounting and Transition Records

The JSON statistics distinguish generated work from committed Markov transitions:

| Field | Definition |
| --- | --- |
| `batch_sizes`, `total_workload` | Refinement suffix requests per batch and in total, including unused speculation |
| `bundle_batch_sizes`, `total_bundles` | Physical bundles of $K$ candidates generated per batch and in total |
| `walked_steps`, `total_walked_steps` | Committed MTM transitions per batch and in total |
| `unused_suffixes` | `total_workload` minus $K$ times `total_walked_steps` |
| `generation_calls`, `scoring_calls`, `total_nfe` | Backend generation calls, additional scoring calls, and their sum, including block extensions |
| `scoring_requests` | Individual token-scoring requests, each producing one discarded vLLM token |
| `generated_tokens` | Actual suffix decoding, including extensions and Hugging Face overgeneration, excluding discarded scoring tokens |
| `proposal_suffix_tokens`, `extension_tokens` | Requested refinement and extension token counts |
| `planned_nodes`, `deduplicated_bundles`, `planner_limit_hits` | Retained logical nodes, bundle allocations avoided by sharing, and node-visit limit events |
| `logprob_sum`, `log_likelihood` | Sum and mean base-model token log probability of the final trimmed response |

Here `total_nfe` counts calls to the backend generation API rather than individual transformer forward passes. For vLLM, the decoded-token count including scoring completions is `generated_tokens + scoring_requests`.

Each transition records its cut, selected candidate, acceptance decision, log weights, and outcome probabilities. `candidate_outcome_probabilities` lists the $K$ candidate outcomes followed by rejection, matching the node's `children` order. Persistent `node_id` values identify nodes within a block, while `tree_index` and `next_tree_index` use the local indices of the current prefetched subtree.

The recorded outcome entropy and effective number of outcomes are

$$
\mathcal{H}_{\mathrm{out}}
=-\sum_{o:\rho_o>0}\rho_o\log\rho_o,
\qquad
N_{\mathrm{eff,out}}=\exp(\mathcal{H}_{\mathrm{out}}),
$$

where the sum includes rejection and every candidate slot. These quantities describe labeled outcomes. Duplicate candidate sequences can reduce the number of distinct next states without reducing the number of labeled outcomes.

## 7. Validation Scope and Limitations

The [shared-engine tests](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_prefetch_multi_try.py) use a normalized autoregressive oracle and independent transition replay across multiple seeds, budgets, candidate counts, blocks, and all seven rank policies. Finite-state enumeration checks normalization and detailed balance for $K\in\{1,2,3\}$. Additional cases cover deferred infeasible cuts, bundle sharing at equal depth, fresh candidates across successive transitions, EOS handling, and work accounting. Seeded reproducibility checks hold the scheduling configuration fixed.

The [tree tests](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_prefetch_multi_try_tree.py) cover candidate and rejection children, outcome probabilities, and state-dependent descendant weights. The [Hugging Face adapter tests](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_hf_subtree_multi_try_mh.py) and [vLLM adapter tests](/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/src/tests/test_vllm_subtree_multi_try_mh.py) use model substitutes to verify request construction, score alignment, context handling, reproducibility, and backend work counts. Runner tests cover configuration validation, dataset batching, and output metadata.

These checks address the implemented transition and scheduling contracts. They do not measure convergence of an LLM chain after a finite refinement budget, nor do they establish that block extension followed by EOS trimming produces an exact draw from a variable-length power distribution. Installed backend integration, numerical behavior, GPU memory usage, and wall-clock performance require GPU experiments. The implementation alone establishes no speedup claim.

## Reference

[1] J. S. Liu, F. Liang, and W. H. Wong, “The Multiple-Try Method and Local Optimization in Metropolis Sampling,” *Journal of the American Statistical Association*, vol. 95, no. 449, pp. 121–134, 2000. [doi:10.1080/01621459.2000.10473908](https://doi.org/10.1080/01621459.2000.10473908).

[2] H. Guo, N. Guo, C. Meinel, and H. Yang, “Sample Where You Struggle: Sharpening Base Model Reasoning via Entropy-Guided Power Sampling,” *arXiv preprint*, arXiv:2606.09926, 2026. [doi:10.48550/arXiv.2606.09926](https://doi.org/10.48550/arXiv.2606.09926). The implementation follows the MTM weighting and acceptance rule in [Section 3.2, Eqs. (6)–(7), version 1](https://arxiv.org/html/2606.09926v1#S3.SS2).

[3] G. Ye and S. Lu, “Prefetching-based Multiproposal Markov Chain Monte Carlo Algorithm,” *IEEE Transactions on Artificial Intelligence*, vol. 5, no. 9, pp. 4493–4505, 2024. [doi:10.1109/TAI.2024.3385384](https://doi.org/10.1109/TAI.2024.3385384). Accompanying source code: [PMP-MCMC](https://github.com/guifengye1/PMP-MCMC), including the [`PMP` comparison implementation](https://github.com/guifengye1/PMP-MCMC/blob/main/simple_sampling/error/error.py).
