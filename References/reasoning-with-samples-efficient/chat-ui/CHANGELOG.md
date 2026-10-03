# Chat UI Updates

## Latest Changes (v2)

### Method-Specific Settings in Columns
- **Moved**: All method-specific parameters from sidebar to individual columns
- **UI Change**: Each method now has a **⚙️ Settings** expander in its column
- **Sidebar**: Now only contains global settings (API key, model, max tokens, method selection)
- **Benefits**:
  - Cleaner sidebar
  - Settings appear next to the method they affect
  - Easier to understand which parameters affect which method
  - Can compare different configurations side-by-side

### Simplified Layout
- Method name appears as column header (from st.subheader)
- Settings in collapsible expander (default: closed)
- Description shown in italics below response
- Response and stats remain in table-like layout

## Previous Changes (v1)

### 1. New Default Sampling Methods
- **Changed from**: Greedy, Temperature, MCMC
- **Changed to**: Greedy, MCMC, Beam Search (default)
- Temperature Sampling is now optional (unchecked by default)

### 2. Added Beam Search Implementation
- Created `BeamSearchSampling` class in `src/benchmark_runner.py`
- Generates multiple candidates using OpenAI's `n` parameter
- Selects best completion based on average log probability
- Configurable parameters:
  - Number of beams (2-10, default: 4)
  - Beam temperature (0.1-2.0, default: 0.7)

### 3. Table-like Stats Layout
- **Before**: Separate comparison section at the bottom with stats in rows
- **After**: Stats displayed in vertically aligned columns below all responses
- Layout structure:
  - Row 1: Method names + descriptions (side-by-side)
  - Row 2: Responses in scrollable boxes (side-by-side, max-height: 400px)
  - Row 3: Stats in aligned columns (table-like layout)
- Stats include:
  - Token usage (prompt + completion)
  - Generation time
  - Output length
  - MCMC-specific: Acceptance ratio
- No white background boxes - clean, minimal design

### 4. Added Method Descriptions
- Each method now displays a concise description explaining how it works:
  - **Greedy Decoding**: Selects most probable token, deterministic
  - **MCMC Power Sampling**: Metropolis-Hastings with accept/reject
  - **Beam Search**: Multiple parallel hypotheses with best selection
  - **Temperature Sampling**: Controlled randomness via logit scaling

### 5. Fixed MCMC Edge Case
- **Issue**: "This should not happen" assertion when insufficient blocks
- **Fix**: Added proper check to skip MCMC refinement when `num_complete_blocks < 1`

### 6. Parallel Execution (Maintained)
- All selected methods run simultaneously
- Results populate as each completes
- Shows real-time timing comparisons

## Files Modified

1. **src/benchmark_runner.py**
   - Added `BeamSearchSampling` class
   - Fixed MCMC edge case handling

2. **chat-ui/app.py**
   - Updated default selections (Greedy, MCMC, Beam Search)
   - Added beam search configuration UI
   - Transposed stats to appear under each response
   - Added method descriptions
   - Maintained parallel execution

3. **chat-ui/README.md**
   - Updated sampling methods section
   - Added beam search documentation
   - Updated usage instructions

4. **README.md** (main)
   - Updated feature list to reflect new defaults
   - Added mention of method descriptions

## Usage

```bash
streamlit run chat-ui/app.py
```

Default configuration compares:
- Greedy Decoding (temperature=0)
- MCMC Power Sampling (α=4.0, 10 steps, block=192)
- Beam Search (4 beams, temperature=0.7)

All run in parallel with results appearing as they complete!
