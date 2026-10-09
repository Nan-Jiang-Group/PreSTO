"""Power-SMC on stock vLLM, with base-model log p from a prompt-logprob rescoring pass.

Run with the installed package:
    python -m power_sharpening.runners.vllm.run_standard_smc --dataset lcb_v6 --algorithm smc \
        --override n_particles=16 max_samples=20 batch_size=1

Same CLI, SMC loop, timing line (``smc_power_sampler took <s> seconds for <i>-th prompts``), resource probe
(KV-cache occupancy, eviction, prefix-cache hits, GPU memory in the CSV and ``<run>.resources.json``), and output
format as ``run_power_smc.py``; only the engine differs. ``smc_score_batch_size`` sets how many particles one
rescoring call takes.
"""

from power_sharpening.runners.vllm.run_power_smc import main


if __name__ == "__main__":
    main(engine="standard")
