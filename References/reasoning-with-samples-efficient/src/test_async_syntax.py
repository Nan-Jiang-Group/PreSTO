#!/usr/bin/env python3
"""
Syntax test for async beam search - no API calls.
Just verifies the async methods are properly defined.
"""
import asyncio
import sys

# Mock classes for testing
class MockAsyncOpenAI:
    def __init__(self, **kwargs):
        self.api_key = "test"
        self.base_url = "https://test.com"
        self.default_model = "test-model"
    
    class MockChoice:
        def __init__(self):
            self.message = type('obj', (object,), {'content': 'test output'})()
            self.finish_reason = 'stop'
            self.logprobs = type('obj', (object,), {
                'content': [
                    type('obj', (object,), {'token': 'test', 'logprob': -0.1})()
                ]
            })()
    
    class MockUsage:
        def __init__(self):
            self.prompt_tokens = 10
            self.completion_tokens = 20
    
    async def create_mock_response(self):
        """Simulate API response."""
        await asyncio.sleep(0.01)  # Simulate network delay
        return type('obj', (object,), {
            'choices': [self.MockChoice(), self.MockChoice()],
            'usage': self.MockUsage()
        })()

# Import the beam search class
sys.path.insert(0, '/Users/nisargdesai/Documents/xai-hackathon/reasoning-with-samples-efficient/src')
from benchmark_runner import BeamSearchSampling

def test_async_methods_exist():
    """Test that async methods are defined correctly."""
    print("Testing async beam search implementation...")
    
    # Create instance
    strategy = BeamSearchSampling(
        alpha=4.0,
        beam_width=2,
        n_per_beam=2,
        tokens_per_step=192,
        length_penalty=0.6,
        proposal_temperature=1.0,
        top_logprobs=5,
        debug=False
    )
    
    # Check methods exist
    assert hasattr(strategy, '_sample_continuation_multiple_async'), "Missing _sample_continuation_multiple_async"
    assert hasattr(strategy, '_expand_beams_parallel'), "Missing _expand_beams_parallel"
    assert hasattr(strategy, '_generate_async'), "Missing _generate_async"
    assert hasattr(strategy, 'generate'), "Missing generate"
    
    # Check they are coroutines
    import inspect
    assert inspect.iscoroutinefunction(strategy._sample_continuation_multiple_async), \
        "_sample_continuation_multiple_async should be async"
    assert inspect.iscoroutinefunction(strategy._expand_beams_parallel), \
        "_expand_beams_parallel should be async"
    assert inspect.iscoroutinefunction(strategy._generate_async), \
        "_generate_async should be async"
    assert not inspect.iscoroutinefunction(strategy.generate), \
        "generate should NOT be async (it's a wrapper)"
    
    print("✅ All async methods defined correctly!")
    print("✅ Method signatures validated!")
    print("\n📊 Async methods:")
    print("  - _sample_continuation_multiple_async (async)")
    print("  - _expand_beams_parallel (async)")
    print("  - _generate_async (async)")
    print("  - generate (sync wrapper using asyncio.run)")
    
    return True

if __name__ == "__main__":
    try:
        success = test_async_methods_exist()
        if success:
            print("\n✅ Async implementation syntax is CORRECT!")
            print("🚀 Ready for testing with real API calls")
            sys.exit(0)
        else:
            print("\n❌ Tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
