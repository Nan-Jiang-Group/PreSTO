import os
import sys
import streamlit as st
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import base64

# Add parent directory to path to import from src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.benchmark_runner import (
    GreedySampling,
    MCMCSampling,
    ParallelMCMCSampling,
    TemperatureSampling,
    BeamSearchSampling,
)
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Reasoning with Sampling - Interactive Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .method-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stats {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🧠 Reasoning with Sampling</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Compare different sampling strategies side-by-side</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Configuration
    st.subheader("API Settings")
    
    # X.AI / Grok configuration
    xai_api_key = st.text_input(
        "X.AI API Key (for Grok)",
        value=os.getenv("XAI_API_KEY", ""),
        type="password",
        help="Get your API key from https://console.x.ai/"
    )
    
    grok_model = st.selectbox(
        "Grok Model",
        ["grok-4-1-fast-non-reasoning", "grok-2-1212", "grok-beta", "grok-2-latest", "grok-4-1-fast-reasoning"],
        index=0,
        help="Grok model for greedy sampling"
    )
    
    st.divider()
    
    # RunPod / Qwen configuration
    runpod_endpoint = st.text_input(
        "RunPod Endpoint URL (for Qwen)",
        value=os.getenv("RUNPOD_ENDPOINT", "https://44dss5ov9819bi-8000.proxy.runpod.net/v1"),
        help="Your RunPod vLLM endpoint URL"
    )
    
    qwen_model = st.text_input(
        "Qwen Model Name",
        value="Qwen/Qwen2.5-7B",
        help="Model name on RunPod"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=50,
        max_value=4096,
        value=4096,
        step=50,
        help="Maximum number of tokens to generate"
    )
    
    st.divider()
    
    # Sampling Methods Selection
    st.subheader("Sampling Methods")
    
    use_greedy = st.checkbox("Greedy Decoding", value=True)
    use_mcmc = st.checkbox("MCMC Power Sampling", value=True)
    
    # MCMC variant selection (only shown if MCMC is enabled)
    if use_mcmc:
        mcmc_variant = st.radio(
            "MCMC Variant",
            ["Parallel MCMC (Faster)", "Serial MCMC"],
            index=0,  # Default to Parallel
            help="Parallel MCMC generates multiple proposals simultaneously (faster)"
        )
    else:
        mcmc_variant = "Parallel MCMC (Faster)"  # Default when not shown
    
    use_beam_search = st.checkbox("Beam Search", value=True)
    use_temperature = st.checkbox("Temperature Sampling", value=False)
    
    st.divider()
    st.markdown("""
    ### About
    This demo compares different sampling strategies from the paper 
    [**Reasoning with Sampling**](https://arxiv.org/abs/2510.14901).
    
    **Sampling Methods:**
    - **Greedy**: Deterministic (temp=0)
    - **MCMC**: Power sampling with Calderhead (Parallelizable Metropolis-Hastings)
      - **Parallel** (default): Multiple proposals simultaneously (faster)
      - **Serial**: One proposal at a time (slower but simpler)
    - **Beam Search**: Best-first search across multiple candidates
    - **Temperature**: Standard sampling (optional)
    
    **Providers:**
    - **X.AI**: Grok models via X.AI API
    - **RunPod**: Self-hosted Qwen 3.8B on RunPod
    - **vLLM (Local)**: Local vLLM server
    - **Ollama**: Local Ollama server
    """)

# Main content area
st.header("💬 Prompt")
prompt = st.text_area(
    "Enter your prompt",
    height=150,
    placeholder="Example: Write a Python function to compute the Fibonacci sequence...",
    help="Enter the prompt you want to test with different sampling methods"
)

# Run button
run = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = {}

if run and prompt.strip():
    # Validation
    if not xai_api_key:
        st.error("⚠️ Please enter your X.AI API key in the sidebar")
    elif not runpod_endpoint:
        st.error("⚠️ Please enter the RunPod endpoint URL")
    else:
        # Initialize clients
        grok_client = OpenAI(api_key=xai_api_key, base_url="https://api.x.ai/v1")
        grok_client.default_model = grok_model
        
        qwen_client = OpenAI(api_key="runpod", base_url=runpod_endpoint)
        qwen_client.default_model = qwen_model

        # Display provider info
        st.info(f"🔧 Using Grok ({grok_model}) + Qwen ({qwen_model} at {runpod_endpoint})")

        # Fixed 4-column layout with specific strategies
        strategy_configs = [
            ("Greedy Grok", "greedy_grok", grok_client),
            ("Greedy Qwen", "greedy_qwen", qwen_client),
            ("Parallel MCMC Qwen", "mcmc_parallel", qwen_client),
            ("Beam Search Qwen", "beam_search", qwen_client),
        ]
        
        st.header("📊 Results")
            
        # Create columns for side-by-side comparison
        cols = st.columns(len(strategy_configs))
            
        # Create placeholders for each column and collect method-specific settings
        placeholders = []
        stats_containers = []
        strategies = []
            
        for idx, (col, (name, method_type, client)) in enumerate(zip(cols, strategy_configs)):
            with col:
                st.subheader(name)
                    
                # Method-specific settings in expander
                with st.expander("⚙️ Settings", expanded=False):
                    if method_type == "mcmc":
                        alpha = st.slider(
                            "Alpha (α)",
                            min_value=1.0,
                            max_value=8.0,
                            value=4.0,
                            step=0.5,
                            help="Power for target distribution p^α",
                            key=f"alpha_{idx}"
                        )
                            
                        mcmc_steps = st.slider(
                            "MCMC Steps",
                            min_value=1,
                            max_value=20,
                            value=10,
                            step=1,
                            help="Number of MCMC refinement steps per block",
                            key=f"mcmc_steps_{idx}"
                        )
                            
                        block_size = st.slider(
                            "Block Size",
                            min_value=32,
                            max_value=512,
                            value=192,
                            step=32,
                            help="Block size for block-wise generation",
                            key=f"block_size_{idx}"
                        )
                            
                        proposal_temperature = st.slider(
                            "Proposal Temperature",
                            min_value=0.1,
                            max_value=2.0,
                            value=1.0,
                            step=0.1,
                            help="Temperature for MCMC proposal distribution",
                            key=f"proposal_temp_{idx}"
                        )
                            
                        debug_mcmc = st.checkbox("Debug MCMC", value=False, key=f"debug_{idx}")
                            
                        strategy = MCMCSampling(
                            alpha=alpha,
                            mcmc_steps=mcmc_steps,
                            block_size=block_size,
                            proposal_temperature=proposal_temperature,
                            debug=debug_mcmc
                        )
                        strategy_name = f"Serial MCMC (α={alpha}, steps={mcmc_steps})"
                        
                    elif method_type == "mcmc_parallel":
                        alpha = st.slider(
                            "Alpha (α)",
                            min_value=1.0,
                            max_value=8.0,
                            value=4.0,
                            step=0.5,
                            help="Power for target distribution p^α",
                            key=f"alpha_{idx}"
                        )
                            
                        mcmc_steps = st.slider(
                            "MCMC Steps",
                            min_value=1,
                            max_value=10,
                            value=1,
                            step=1,
                            help="Number of MCMC refinement steps per block (1 is typical for parallel)",
                            key=f"mcmc_steps_{idx}"
                        )
                            
                        block_size = st.slider(
                            "Block Size",
                            min_value=16,
                            max_value=256,
                            value=32,
                            step=16,
                            help="Block size for block-wise generation",
                            key=f"block_size_{idx}"
                        )
                            
                        num_proposals = st.slider(
                            "Number of Proposals",
                            min_value=2,
                            max_value=20,
                            value=10,
                            step=1,
                            help="Number of parallel proposals per MCMC step",
                            key=f"num_proposals_{idx}"
                        )
                            
                        proposal_temperature = st.slider(
                            "Proposal Temperature",
                            min_value=0.1,
                            max_value=2.0,
                            value=0.25,
                            step=0.05,
                            help="Temperature for MCMC proposal distribution",
                            key=f"proposal_temp_{idx}"
                        )
                            
                        debug_mcmc = st.checkbox("Debug MCMC", value=False, key=f"debug_{idx}")
                            
                        strategy = ParallelMCMCSampling(
                            alpha=alpha,
                            mcmc_steps=mcmc_steps,
                            block_size=block_size,
                            num_proposals=num_proposals,
                            proposal_temperature=proposal_temperature,
                            debug=debug_mcmc,
                            api_key="runpod",
                            base_url=runpod_endpoint,
                            model=qwen_model,
                            supports_n_param=True
                        )
                        strategy_name = f"Parallel MCMC (α={alpha}, N={num_proposals})"
                            
                    elif method_type == "beam_search":
                        beam_width = st.slider(
                            "Beam Width",
                            min_value=2,
                            max_value=10,
                            value=5,
                            step=1,
                            help="Number of parallel beams to maintain",
                            key=f"beam_width_{idx}"
                        )
                            
                        alpha = st.slider(
                            "Alpha (α)",
                            min_value=1.0,
                            max_value=8.0,
                            value=4.0,
                            step=0.5,
                            help="Power factor for beam scoring",
                            key=f"beam_alpha_{idx}"
                        )
                            
                        proposal_temperature = st.slider(
                            "Proposal Temperature",
                            min_value=0.1,
                            max_value=2.0,
                            value=0.1,
                            step=0.1,
                            help="Temperature for beam proposals",
                            key=f"beam_proposal_temp_{idx}"
                        )
                            
                        tokens_per_step = st.slider(
                            "Tokens per Step",
                            min_value=32,
                            max_value=512,
                            value=512,
                            step=32,
                            help="Tokens to generate per beam expansion",
                            key=f"tokens_per_step_{idx}"
                        )
                            
                        strategy = BeamSearchSampling(
                            alpha=alpha,
                            beam_width=beam_width,
                            n_per_beam=7,
                            proposal_temperature=proposal_temperature,
                            tokens_per_step=tokens_per_step,
                            supports_n_param=True  # Most providers support this
                        )
                        strategy_name = f"Beam Search (width={beam_width}, α={alpha})"
                            
                    elif method_type == "temperature":
                        temperature = st.slider(
                            "Temperature",
                            min_value=0.1,
                            max_value=2.0,
                            value=0.8,
                            step=0.1,
                            help="Higher = more random, Lower = more deterministic",
                            key=f"temperature_{idx}"
                        )
                            
                        strategy = TemperatureSampling(temperature=temperature)
                        strategy_name = f"Temperature (T={temperature})"
                            
                    else:  # greedy
                        st.info("No configuration needed - uses temperature=0")
                        strategy = GreedySampling()
                        strategy_name = "Greedy Decoding"
                    
                strategies.append((strategy_name, strategy, client))
                    
                placeholders.append({
                    'header': st.empty(),
                    'spinner': st.empty(),
                    'response': st.empty(),
                    'error': st.empty()
                })
            
        # Create a separate row for stats (will be aligned)
        st.markdown("---")
        stats_cols = st.columns(len(strategy_configs))
        image_containers = []
        for col in stats_cols:
            with col:
                stats_containers.append(st.empty())
                # Regular container for images (not empty, so it doesn't get overwritten)
                image_containers.append(col.container())
            
        # Function to run a single strategy
        def run_strategy(name, strategy, client):
            try:
                start_time = time.time()
                completion, prompt_tokens, completion_tokens = strategy.generate(
                    client,
                    prompt,
                    max_tokens=max_tokens
                )
                elapsed_time = time.time() - start_time
                    
                result = {
                    'completion': completion,
                    'tokens': prompt_tokens + completion_tokens,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'time': elapsed_time,
                    'error': None
                }
                    
                # Get MCMC-specific stats
                if isinstance(strategy, MCMCSampling):
                    result['acceptance_ratio'] = strategy.get_acceptance_ratio()
                    
                return name, result
                    
            except Exception as e:
                return name, {'error': str(e)}
            
        # Run all strategies in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            # Submit all tasks
            future_to_strategy = {
                executor.submit(run_strategy, name, strategy, client): (idx, name)
                for idx, (name, strategy, client) in enumerate(strategies)
            }
                
            # Show initial state
            for idx, (name, _, _) in enumerate(strategies):
                placeholders[idx]['header'].subheader(name)
                placeholders[idx]['spinner'].info("⏳ Generating...")
                
            # Process results as they complete
            for future in as_completed(future_to_strategy):
                idx, name = future_to_strategy[future]
                strategy_name, result = future.result()
                    
                # Clear spinner
                placeholders[idx]['spinner'].empty()
                    
                if result.get('error'):
                    # Show error
                    placeholders[idx]['error'].error(f"❌ Error: {result['error']}")
                else:
                    # Method description
                    descriptions = {
                        'Greedy Decoding': 'Selects the most probable token at each step, providing deterministic and focused outputs. Best for tasks requiring consistency.',
                        'MCMC': 'Uses Calderhead (Parallelizable Metropolis-Hastings) to sample from p^α distribution. Explores high-probability regions through iterative refinement with accept/reject steps.',
                        'Beam Search': 'Maintains multiple parallel hypotheses and selects the best based on cumulative probability scores. Balances exploration and exploitation.',
                        'Temperature': 'Introduces controlled randomness by scaling logits. Higher temperature increases diversity; lower increases focus.'
                    }
                        
                    # Find matching description
                    method_desc = None
                    for key in descriptions:
                        if key in strategy_name:
                            method_desc = descriptions[key]
                            break
                        
                    if method_desc:
                        placeholders[idx]['header'].markdown(
                            f"_{method_desc}_",
                            unsafe_allow_html=False
                        )
                        
                    # Display result
                    response_html = f"""
                    <div class="method-card" style="max-height: 400px; overflow-y: auto; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 1rem;">
                        {result["completion"]}
                    </div>
                    """
                    placeholders[idx]['response'].markdown(response_html, unsafe_allow_html=True)
                        
                    # Stats in aligned row below (no background, clean table-like layout)
                    stats_parts = [
                        '<div style="padding: 0.5rem 0;">',
                        f'<div style="margin-bottom: 0.75rem; font-size: 0.95rem;">',
                        f'<strong>📝 Tokens:</strong> {result["tokens"]}',
                        f'<div style="color: #666; font-size: 0.85rem; margin-left: 1.5rem;">',
                        f'prompt: {result["prompt_tokens"]}, completion: {result["completion_tokens"]}',
                        f'</div>',
                        f'</div>',
                        f'<div style="margin-bottom: 0.75rem; font-size: 0.95rem;">',
                        f'<strong>⏱️ Time:</strong> {result["time"]:.2f}s',
                        f'</div>',
                        f'<div style="margin-bottom: 0.75rem; font-size: 0.95rem;">',
                        f'<strong>📏 Length:</strong> {len(result["completion"])} chars',
                        f'</div>',
                    ]
                        
                    # MCMC-specific stats
                    if 'acceptance_ratio' in result:
                        stats_parts.extend([
                            f'<div style="margin-bottom: 0.75rem; font-size: 0.95rem;">',
                            f'<strong>✅ Acceptance:</strong> {result["acceptance_ratio"]:.1%}',
                            f'</div>',
                        ])
                        
                    stats_parts.append('</div>')
                    stats_html = ''.join(stats_parts)
                        
                    stats_containers[idx].markdown(stats_html, unsafe_allow_html=True)
                    
                    # Display method visualization images
                    method_type = strategy_configs[idx][1]
                    if method_type == "mcmc_parallel":
                        image_path = Path(__file__).parent.parent / "public" / "mcmc.png"
                        with open(image_path, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode()
                        with image_containers[idx]:
                            st.markdown(
                                f'<div style="background-color: white; padding: 10px; border-radius: 8px; margin-top: 10px;">'
                                f'<img src="data:image/png;base64,{img_data}" style="width: 100%; height: auto;">'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    elif method_type == "beam_search":
                        image_path = Path(__file__).parent.parent / "public" / "beam-search.svg"
                        with open(image_path, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode()
                        with image_containers[idx]:
                            st.markdown(
                                f'<div style="background-color: white; padding: 10px; border-radius: 8px; margin-top: 10px;">'
                                f'<img src="data:image/svg+xml;base64,{img_data}" style="width: 100%; height: auto;">'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                    results[strategy_name] = result
            
        # Store results in session state
        st.session_state.results = results

elif run and not prompt.strip():
    st.warning("⚠️ Please enter a prompt")

# Show previous results if available
if not run and st.session_state.results:
    st.info("💡 Previous results are still available. Modify settings and run again to compare.")
