Beam Search, Top-k Sampling, and Top-k-Constrained Beam Search (k-CBS)

A practical, implementation-oriented guide with contrasts, math intuition, and a vLLM-friendly plan.

⸻

Executive summary
	•	Top-k sampling is stochastic: at each step it filters to the k highest-probability tokens and renormalizes, then samples one.  ￼
	•	Beam search is deterministic: it keeps B best partial sequences (beams), expands all with every token, and keeps the best B again, repeating until termination.  ￼
	•	Top-k-constrained beam search (k-CBS) ties them together: run beam search with width B = k but only allow top-k tokens per beam at each step; this surfaces high-probability continuations under top-k decoding without sampling. It’s far more efficient than Monte-Carlo top-k sampling when you want the most-likely continuations, not random draws.  ￼  ￼

There are two useful k-CBS variants for downstream probability estimates: (1) use the final k beams at step T, or (2) return the k² candidates before the last prune (at step T−1). Both support lower/upper bounds for a near-verbatim probability p_{z,\varepsilon}.  ￼  ￼

⸻

1) Background and terminology (very short)

A language model maps a token sequence to a distribution over the next token. Decoding schemes transform that distribution before choosing the next token (e.g., temperature scaling, top-k filtering).  ￼

We’ll refer to:
	•	V: vocabulary; T: continuation length;
	•	score(sequence): sum of (possibly transformed) log-probabilities;
	•	distance metrics for near-verbatim comparisons: Hamming (substitutions only) and Levenshtein (insertions/deletions/substitutions).  ￼

⸻

2) Top-k sampling

Definition. At time t, keep only the k tokens with highest probability, renormalize the distribution over those tokens, and sample one as x_{t+1}.  ￼

Properties.
	•	Stochastic (nondeterministic): repeated runs produce different sequences.
	•	Diversity comes from randomness; quality depends on a good choice of k.
	•	Compute per step scales with evaluating the model once and working with k candidates (not |V|).
	•	Great when you want variety of plausible continuations; weaker when you need the top-mass outcomes deterministically.

⸻

3) Beam search (vanilla)

Definition. Keep B partial sequences. At each step:
	1.	Expand each beam by every token (conceptually B\times |V| children).
	2.	Score all children (sum log-probs).
	3.	Prune to the top B.
	4.	Repeat until EOS or length T.  ￼

Properties.
	•	Deterministic (given fixed model/logits transform).
	•	Tends to concentrate on the highest-probability region; diversity can collapse if beams share best branches.
	•	Compute per step is heavier conceptually because of the |V| expansion (implementations use tricks, but the idea stands).

⸻

4) Top-k-constrained beam search (k-CBS)

Key idea. Set beam width B = k and, during expansion, restrict each beam to its top-k next tokens; i.e., mask all but the top-k logits, score, and prune globally to B again. Result: the final k beams at step T are high-probability continuations under top-k decoding, surfaced deterministically.  ￼

Why it’s useful. It returns likely continuations without sampling noise and at far lower cost than Monte-Carlo. For example, for 50-token suffixes under top-40 decoding, a simple top-k Monte-Carlo estimator might need \sim 10^7 samples for a reliable estimate; k-CBS needs only \mathcal{O}(T\cdot k) token evaluations—orders of magnitude cheaper.  ￼  ￼

Two practical variants.
	•	Final-beam variant (k): take the k sequences in the final beam (step T). Filtering them by a distance metric yields a lower bound on the near-verbatim probability p_{z,\varepsilon}.  ￼  ￼
	•	k² pre-prune variant: note that right before the final prune, there are up to k^2 candidates (each of the k beams expands by k tokens). All of those are valid sequences under top-k decoding, so return them instead of pruning at step T{-}1 and use them for tighter bounds.  ￼

Bounding p_{z,\varepsilon}. Both variants compute lower/upper bounds on the probability that the model, under top-k decoding, generates a suffix within distance \varepsilon of the target. Empirically, even the simple baseline gives high-fidelity lower bounds.  ￼

Distance-aware pruning. You can bake the distance metric into the search: prune partial sequences that already exceed the budget \varepsilon. Hamming is monotone (safe to prune early); Levenshtein isn’t monotone and needs care.  ￼  ￼

⸻

5) Side-by-side comparison

Axis	Top-k sampling	Beam search (B)	Top-k-constrained beam (k-CBS, B=k)
Determinism	❌ (random)	✅	✅
Diversity control	via randomness (k)	limited (B)	moderate (k), but deterministic
Next-token rule	filter to top-k, sample	consider all tokens (conceptually), keep top B	per-beam top-k mask, keep top k
What you get	random draws from top-k process	top-B by (transformed) prob	top sequences under top-k, enumerated
Good for	creative variety	best single answer(s)	efficient surfacing of high-probability continuations under top-k
Typical cost (per step)	\mathcal{O}(k) post-logits ops	$begin:math:text$\mathcal{O}(B\cdot	V


⸻

6) Computing probabilities under top-k for candidates

Given a candidate sequence y = (y_1,\dots,y_T) and a prefix x, the top-k probability of y factors as a product over steps, renormalized within the top-k set at each step. Practically:
	1.	For each step t, identify the top-k tokens under p(\cdot\mid x, y_{<t}).
	2.	If y_t is not in that set, p_k(y_t\mid \cdot)=0; otherwise renormalize the logits/probs over just those k tokens and take the log-prob of y_t.
	3.	Sum over t to get \log p_k(y\mid x), and sum over all candidates within distance \le \varepsilon to form a lower bound on p_{z,\varepsilon}. (Using k² candidates tightens bounds further.)  ￼  ￼

⸻

7) How to implement k-CBS today (using vLLM v0.11.0)

vLLM’s public BeamSearchParams doesn’t expose top_k/top_p constraints. So implement k-CBS in user space by using vLLM as a batched next-token oracle that returns top-k logprobs per prefix; then do the beam bookkeeping yourself.

7.1 High-level algorithm (pseudo-code)

# Inputs: model, prefix x, continuation length T, k (and optionally ε, distance metric)
# Output: final_beam (k sequences) or preprune_candidates (≤k^2 at last step)

beams = [(x_tokens, 0.0)]  # (tokens, logp_k); start with the prefix
for t in range(T):
    # 1) Query model for top-k next-token logprobs for each active beam
    #    Use generate(..., max_tokens=1, logprobs=k, temperature=1.0) per prefix
    expansions = []
    for (tokens, logp_acc) in beams:
        next_topk = topk_next_token_logprobs(tokens, k)  # [(tok, logp, renorm_in_topk)]
        for tok, logp_k in next_topk:
            child = (tokens + [tok], logp_acc + logp_k)
            if within_distance_budget(child, ε):          # optional pruning (Hamming/Levenshtein)
                expansions.append(child)

    # 2) Global prune: keep the top-k by accumulated logp
    if t < T - 1:
        beams = top_k(expansions, k)
    else:
        # Variant A: final k beams
        final_beam = top_k(expansions, k)
        # Variant B: keep ALL expansions here (≤ k^2) before the final prune
        preprune_candidates = expansions

Notes.
	•	The call topk_next_token_logprobs means: request the top-k next tokens and their log-probs for each prefix, and then renormalize over those k to obtain the top-k log-prob mass per token (that is the probability under top-k decoding at that step). The manuscript’s definition of top-k decoding underlies this step.  ￼
	•	Variant choice: return either the final k beams or the k² pre-prune set from the last step. The latter gives tighter bounds later.  ￼
	•	Distance-aware pruning: If you pick Hamming, you can safely prune as you go (it’s monotone non-decreasing); Levenshtein may require more careful bookkeeping.  ￼

7.2 Turning candidates into p_{z,\varepsilon} bounds
	1.	Filter candidates whose distance to the target suffix z is \le \varepsilon.
	2.	Lower bound: sum their top-k sequence probabilities (using the per-step renormalization above).  ￼
	3.	Upper bound: combine the final-beam (k) and pre-prune (k²) sets per the manuscript’s recipe (include all valid top-k sequences discovered up to that frontier).  ￼

7.3 Practical tips
	•	EOS handling. If EOS is within the step’s top-k set, treat it as a valid child; keep beams that end early and pad/stop as needed (the probability mass is then concentrated on EOS at that step).
	•	Length control. If you use a length penalty or prefer exact length T, ensure you score consistently across candidates (e.g., forbid EOS before T if you need fixed-length suffixes).
	•	Batching. Query all active beams in a single batched call each step to amortize overhead.
	•	De-duplication. Different paths can collapse to identical sequences; merge and max-score duplicates.

⸻

8) When to use which?
	•	Choose top-k sampling when diversity and variety are primary—e.g., creative text, brainstorming.
	•	Choose beam search when you want the single best (or top few) sequences under your scoring rule.
	•	Choose k-CBS when you want deterministic, efficient access to the most-likely outcomes under top-k, or when you need tight probability bounds for near-verbatim analysis without running massive Monte-Carlo experiments.  ￼

⸻

9) Appendix: distance metrics (quick reference)
	•	Hamming: counts positional mismatches; good when candidate and target share length T. Monotone, enabling early pruning.  ￼  ￼
	•	Levenshtein: allows insertions/deletions/substitutions; not monotone across steps, so pruning is trickier.  ￼  ￼

⸻

10) Key takeaways
	•	Beam search ≠ sampling; it enumerates and prunes.
	•	Top-k sampling ≠ beam; it renormalizes and samples in the k-set.
	•	k-CBS connects them cleanly: B=k + per-beam top-k constraint gives you deterministic top-k outcomes of the sampling process, often at dramatically lower cost than Monte-Carlo when you care about the likely region.  ￼  ￼

⸻

Citations to the manuscript you shared
	•	Relationship between top-k and beam (B=k), and the k-mask scoring rule:  ￼
	•	Efficiency vs Monte-Carlo (orders of magnitude fewer token evals):  ￼  ￼
	•	Two variants (final k vs k² pre-prune) and bounds on p_{z,\varepsilon}:  ￼  ￼
	•	Distance metrics and monotonicity remarks:  ￼  ￼

⸻

If you want, I can turn the pseudo-code into a ready-to-run Python module that plugs into vLLM (batched, with Hamming/Levenshtein options and k/k² output modes).