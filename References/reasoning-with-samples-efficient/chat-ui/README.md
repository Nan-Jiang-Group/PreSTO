# Chat UI - Interactive Sampling Comparison

An interactive Streamlit application for comparing different sampling strategies side-by-side.

## Features

- 🎯 **Side-by-side comparison** of multiple sampling methods
- ⚡ **Parallel execution** - all methods run simultaneously for faster results
- 🔧 **Configurable parameters** for each sampling strategy
- 📊 **Real-time statistics** including token usage, timing, and acceptance rates
- 📋 **Table-like layout** with vertically aligned stats for easy comparison
- 🎨 **Clean, responsive UI** with easy-to-use controls

## Sampling Methods (Default: Greedy, MCMC, Beam Search)

1. **Greedy Decoding** (Default): Deterministic output (temperature=0)
   - Selects most probable token at each step
   - Best for consistency and reproducibility

2. **MCMC Power Sampling** (Default): Advanced sampling using Metropolis-Hastings algorithm
   - Configurable alpha (α) for power distribution p^α
   - Adjustable MCMC steps and block size
   - Real-time acceptance rate tracking
   - Explores high-probability regions through refinement

3. **Beam Search** (Default): Best-first search across multiple candidates
   - Configurable number of beams (2-10)
   - Selects best output based on cumulative probability
   - Balances exploration and exploitation

4. **Temperature Sampling** (Optional): Standard sampling with configurable temperature
   - Higher = more random, Lower = more deterministic

## Setup

### Prerequisites

Make sure you have the project dependencies installed:

```bash
# From the project root
uv pip install -r pyproject.toml
```

Or install Streamlit separately:

```bash
pip install streamlit
```

### API Key

You'll need an X.AI API key:

1. Get your API key from [X.AI Console](https://console.x.ai/)
2. Create a `.env` file in the project root:

```bash
echo "XAI_API_KEY=your-api-key-here" > ../.env
```

Or enter it directly in the UI sidebar.

## Running the App

From the project root directory:

```bash
streamlit run chat-ui/app.py
```

Or from the `chat-ui` directory:

```bash
cd chat-ui
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

1. **Configure Global Settings** (sidebar):
   - Enter your X.AI API key
   - Select the model (e.g., grok-2-1212)
   - Set max tokens
   - Choose which methods to compare (Greedy, MCMC, Beam Search by default)

2. **Enter Your Prompt**:
   - Type your prompt in the text area
   - Example: "Write a Python function to compute Fibonacci numbers"

3. **Adjust Method-Specific Settings** (optional):
   - Each method column has a **⚙️ Settings** expander
   - Expand to tune parameters for that specific method
   - Changes apply when you click "Run Comparison"

4. **Run Comparison**:
   - Click "🚀 Run Comparison"
   - All methods execute in parallel
   - Each column shows:
     - Method name and description
     - Settings expander
     - Response in scrollable box (max height: 400px)
   - Stats appear aligned below all responses (table-like layout)
   - Compare token usage, timing, and output quality in real-time

## Configuration Options

### Global Settings (Sidebar)
- **API Key**: Your X.AI API key
- **Model**: Choose from available models (grok-2-1212, grok-beta, etc.)
- **Max Tokens**: Maximum tokens to generate (50-2048)
- **Method Selection**: Check/uncheck which methods to compare

### Method-Specific Settings (In Each Column)

Each method has its own **⚙️ Settings** expander in its column:

#### Greedy Decoding
- No configuration needed (uses temperature=0)

#### MCMC Power Sampling
- **Alpha (α)**: Power for target distribution (1.0-8.0, default: 4.0)
- **MCMC Steps**: Refinement iterations per block (1-20, default: 10)
- **Block Size**: Tokens per generation block (32-512, default: 192)
- **Proposal Temperature**: Temperature for proposals (0.1-2.0, default: 1.0)
- **Debug Mode**: Show detailed MCMC logs

#### Beam Search
- **Number of Beams**: Parallel sequences to maintain (2-10, default: 4)
- **Beam Temperature**: Temperature for beam sampling (0.1-2.0, default: 0.7)

#### Temperature Sampling (Optional)
- **Temperature**: Controls randomness (0.1-2.0, default: 0.8)

## Example Prompts

### Code Generation
```
Write a Python function to check if a number is prime.
Include docstring and example usage.
```

### Problem Solving
```
Solve this math problem step by step:
A train travels 120 km in 2 hours. 
If it maintains the same speed, how far will it travel in 5 hours?
```

### Creative Writing
```
Write a short story about a robot learning to paint.
```

## Tips

- Start with a simple prompt to test the setup
- Use Greedy + one other method to see the difference clearly
- Try different alpha values (2-6) for MCMC to see how it affects output quality
- Enable MCMC debug mode to understand the sampling process
- Compare token usage across methods to optimize cost
- All methods run in parallel, so you'll see results appear as they complete (faster methods finish first!)

## Troubleshooting

### "Module not found" errors
Make sure you're running from the project root or the parent directory is in Python path.

### API Key errors
Verify your API key is correct and has credits available at [X.AI Console](https://console.x.ai/)

### Slow generation
- Reduce max_tokens
- Reduce mcmc_steps for MCMC sampling
- Try the faster model: grok-4-1-fast-non-reasoning

## Learn More

- [Paper: Reasoning with Sampling](https://arxiv.org/abs/2510.14901)
- [Project Page](https://aakaran.github.io/reasoning_with_sampling/)
- [Main README](../README.md)
