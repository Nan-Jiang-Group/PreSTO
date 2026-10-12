# Reasoning with Sampling

<img width="4480" height="2564" alt="CleanShot 2025-12-07 at 10 21 10@2x" src="https://github.com/user-attachments/assets/69143c2d-e641-4688-b75b-bbb87f5bcaae" />
<img width="5042" height="2780" alt="CleanShot 2025-12-07 at 10 20 57@2x" src="https://github.com/user-attachments/assets/9a57d92e-02a9-445c-941a-60946ac74133" />


### [Paper](https://arxiv.org/abs/2510.14901) | [Project Page](https://aakaran.github.io/reasoning_with_sampling/)

[![rws](teaser.png)](teaser.png)


This repo contains the official PyTorch implementation of Reasoning with Sampling.
> [**Reasoning with Sampling: Your Base Model is Smarter Than You Think**](https://arxiv.org/abs/2510.14901)<br>
> [Aayush Karan](https://aakaran.github.io/), [Yilun Du](https://yilundu.github.io/)
> <br>Harvard<br>



## Setup

```bash
git clone https://github.com/aakaran/reasoning-with-sampling.git
cd reasoning-with-sampling
uv pip install -r pyproject.toml
```

## Interactive Demo (NEW!)

Try out different sampling strategies with our interactive Streamlit app:

```bash
streamlit run chat-ui/app.py
```

Features:
- 🎯 Side-by-side comparison of Greedy, MCMC, and Beam Search (default)
- ⚡ Parallel execution - all methods run simultaneously
- ⚙️ Method-specific settings in each column (not in sidebar!)
- 📊 Token usage, timing, and acceptance rate tracking
- 📋 Table-like layout with vertically aligned stats
- 📝 Method descriptions for each sampling strategy

See [chat-ui/README.md](chat-ui/README.md) for detailed instructions.


## Sampling
The llm_experiments folder contains slurm scripts to run power sampling for MATH500 (```power_samp_math.py```), whose .json is included in llm_experiments/data, as well as HumanEval (```power_samp_he.py```), GPQA Diamond (```power_samp_gpqa.py```), and AlpacaEval 2.0 (```power_samp_alpaca.py```), whose corresponding eval sets can be downloaded from their official repos. 

To run power sampling on MATH500 with 8 seeds and the eval set split across 5 shards:
```bash
sbatch llm_experiments/scripts/power_samp_math.sh
```
The output is several .csv files (based on the shard and seed number) that store the response outputs, correct answers, original prompts, etc. 

## Evaluation
**Single-shot Reasoning**

To grade the responses for single-shot reasoning, collect the .csv files for a given seed run in a folder (e.g. ```results/qwen_math/MATH```) and pass it into ```eval_math.py```:

```bash
python llm_experiments/eval_math.py --folder=results/qwen_math/MATH
```

```eval_gpqa.py``` is similar, and for ```eval_he.py```, an additional ```--output_fname``` argument is required, as HumanEval collects all responses in a jsonl file (e.g. ```--output_fname=qwen_math_he```).

For AlpacaEval 2.0, ```eval_alpaca.py``` collects a ```--folder``` into one json file ```--output_fname```. For evaluating the json file, follow the instructions in the official repo: https://github.com/tatsu-lab/alpaca_eval


**Pass@k Performance**

For pass@k performance, collect the .csv files across seeds in a folder again (e.g. ```results/qwen_math/MATH```) and pass into ```passk_math.py```:
```bash
python llm_experiments/passk_math.py --folder=results/qwen_math/MATH
```
The output is a plot of the pass@k performance. As with single-shot reasoning, ```eval_gpqa.py``` and ```eval_he.py``` are similar, but for the latter an additional ```--output_fname``` argument is required.


