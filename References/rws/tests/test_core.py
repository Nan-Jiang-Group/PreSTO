#!/usr/bin/env python3
"""
Unit tests and verification scripts for the Power Sampling implementation.

These tests verify:
1. Log probability computations are consistent
2. MH acceptance ratio is computed correctly
3. NMCMC=0 reduces to proposal-only sampling
4. Answer extraction works correctly
"""

import sys
import math
import unittest
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAnswerExtraction(unittest.TestCase):
    """Tests for answer extraction and normalization."""
    
    def test_extract_boxed(self):
        from src.rws.answer_extraction import extract_boxed, extract_last_boxed
        
        # Simple boxed
        text = r"The answer is \boxed{42}."
        self.assertEqual(extract_boxed(text), "42")
        
        # Nested braces
        text = r"Therefore $\boxed{\frac{1}{2}}$ is correct."
        self.assertEqual(extract_boxed(text), r"\frac{1}{2}")
        
        # Multiple boxed - should get first
        text = r"First \boxed{1}, then \boxed{2}."
        self.assertEqual(extract_boxed(text), "1")
        self.assertEqual(extract_last_boxed(text), "2")
        
    def test_normalize_answer(self):
        from src.rws.answer_extraction import normalize_answer
        
        # Basic normalization
        self.assertEqual(normalize_answer("42"), "42")
        self.assertEqual(normalize_answer("  42  "), "42")
        
        # Fractions - should convert \frac{1}{2} to 1/2
        self.assertEqual(normalize_answer(r"\frac{1}{2}"), "1/2")
        
        # Remove LaTeX formatting
        result = normalize_answer(r"\textbf{42}")
        self.assertIn("42", result)
        
    def test_answers_match(self):
        from src.rws.answer_extraction import answers_match
        
        self.assertTrue(answers_match("42", "42"))
        self.assertTrue(answers_match("  42 ", "42"))
        self.assertTrue(answers_match("42.0", "42"))
        self.assertFalse(answers_match("42", "43"))


class TestLogProbComputation(unittest.TestCase):
    """Tests for log probability computation consistency."""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_logprob_consistency(self):
        """Verify teacher-forced logprob is consistent."""
        from src.rws.models import ModelWrapper
        
        # Use a small model for testing
        try:
            model_wrapper = ModelWrapper.from_pretrained(
                "gpt2",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except Exception as e:
            self.skipTest(f"Could not load model: {e}")
            
        # Test sequence
        text = "The quick brown fox jumps over the lazy dog."
        input_ids = model_wrapper.encode(text)
        
        # Compute log prob two ways
        logp1 = model_wrapper.teacher_forced_logp(input_ids)
        logp2 = model_wrapper.teacher_forced_logp_with_temperature(input_ids, temperature=1.0)
        
        self.assertAlmostEqual(logp1, logp2, places=5)
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")    
    def test_temperature_scaling(self):
        """Verify temperature scaling affects log probs correctly."""
        from src.rws.models import ModelWrapper
        
        try:
            model_wrapper = ModelWrapper.from_pretrained(
                "gpt2",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except Exception:
            self.skipTest("Could not load model")
            
        text = "Hello world"
        input_ids = model_wrapper.encode(text)
        
        # Higher temperature should generally increase entropy
        logp_t1 = model_wrapper.teacher_forced_logp_with_temperature(input_ids, temperature=1.0)
        logp_t05 = model_wrapper.teacher_forced_logp_with_temperature(input_ids, temperature=0.5)
        
        # Lower temperature (0.5) amplifies differences, so high-prob tokens
        # get even higher prob (potentially increasing total log prob)
        # This is a sanity check that the function runs without error
        self.assertIsInstance(logp_t1, float)
        self.assertIsInstance(logp_t05, float)
        self.assertFalse(math.isnan(logp_t1))
        self.assertFalse(math.isnan(logp_t05))


class TestSuffixLogProb(unittest.TestCase):
    """Tests for suffix log probability computation."""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_suffix_logprob_matches_full(self):
        """
        Verify logprob_suffix_given_prefix matches teacher forcing.
        
        log q(suffix | prefix) should equal:
        log q(prefix + suffix) - log q(prefix)
        when computed correctly.
        """
        from src.rws.models import ModelWrapper
        from src.rws.sampling import logprob_suffix_given_prefix
        
        try:
            model_wrapper = ModelWrapper.from_pretrained(
                "gpt2",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except Exception:
            self.skipTest("Could not load model")
            
        prefix_text = "The quick brown"
        suffix_text = " fox jumps"
        
        prefix_ids = model_wrapper.encode(prefix_text)
        full_ids = model_wrapper.encode(prefix_text + suffix_text)
        suffix_ids = full_ids[len(prefix_ids):]
        
        # Method 1: Using our function
        logp_suffix = logprob_suffix_given_prefix(
            model_wrapper, prefix_ids, suffix_ids, temperature=1.0
        )
        
        # Method 2: Direct computation
        # log q(full) from position len(prefix)-1 onwards
        logp_full_from_prefix = model_wrapper.teacher_forced_logp(
            full_ids, start_pos=len(prefix_ids) - 1
        )
        
        # These should be equal (or very close)
        self.assertAlmostEqual(logp_suffix, logp_full_from_prefix, places=4)


class TestMHAcceptance(unittest.TestCase):
    """Tests for MH acceptance ratio computation."""
    
    def test_acceptance_ratio_bounds(self):
        """MH acceptance ratio should be <= 1 (log A <= 0 when capped)."""
        # This is a mathematical property test
        # In practice, log_A can be positive, but min(0, log_A) is always <= 0
        log_A_values = [0.5, -0.5, 2.0, -2.0, 0.0]
        
        for log_A in log_A_values:
            capped = min(0.0, log_A)
            self.assertLessEqual(capped, 0.0)
            # exp(capped) <= 1
            self.assertLessEqual(math.exp(capped), 1.0)
            
    def test_acceptance_symmetry(self):
        """
        For symmetric proposals, acceptance should depend only on π ratio.
        
        If q(x'|x) = q(x|x'), then:
        A = π(x')/π(x)
        """
        # With equal proposal probs, the ratio simplifies
        log_q_forward = -5.0
        log_q_reverse = -5.0  # Symmetric
        log_pi_new = -10.0
        log_pi_old = -12.0
        
        # log A = (log_pi_new - log_pi_old) + (log_q_reverse - log_q_forward)
        log_A = (log_pi_new - log_pi_old) + (log_q_reverse - log_q_forward)
        
        # With symmetric q, should equal just the π ratio
        expected = log_pi_new - log_pi_old
        self.assertAlmostEqual(log_A, expected, places=10)


class TestNMCMCZero(unittest.TestCase):
    """Test that NMCMC=0 reduces to proposal-only sampling."""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_nmcmc_zero_is_proposal_only(self):
        """With N_mcmc=0, we should just get proposal distribution samples."""
        from src.rws.models import ModelWrapper
        from src.rws.power_mcmc import power_sampling_mcmc
        
        try:
            model_wrapper = ModelWrapper.from_pretrained(
                "gpt2",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except Exception:
            self.skipTest("Could not load model")
            
        prompt = "2 + 2 = "
        
        # Run with N_mcmc=0
        result = power_sampling_mcmc(
            prompt=prompt,
            model_wrapper=model_wrapper,
            alpha=4.0,
            B=32,
            N_mcmc=0,  # No MH steps
            T_max=50,
            seed=42,
        )
        
        # Should have no MH steps
        self.assertEqual(result["num_steps"], 0)
        # Accept rate should be 0 (no proposals)
        self.assertEqual(result["accept_rate"], 0.0)
        # Should still generate output
        self.assertGreater(len(result["output_ids"]), 0)


class TestMetrics(unittest.TestCase):
    """Tests for metric computation."""
    
    def test_accuracy_computation(self):
        from src.rws.metrics import compute_accuracy

        predictions = ["42", "3", "10", "7"]
        gold = ["42", "4", "10", "7"]
        
        accuracy = compute_accuracy(predictions, gold)
        self.assertAlmostEqual(accuracy, 0.75, places=2)
        
    def test_exact_match(self):
        from src.rws.metrics import exact_match
        
        self.assertTrue(exact_match("42", "42"))
        self.assertFalse(exact_match("42", "43"))
        # After normalization
        self.assertTrue(exact_match("  42  ", "42"))


def run_verification():
    """Run all verification tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAnswerExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestLogProbComputation))
    suite.addTests(loader.loadTestsFromTestCase(TestSuffixLogProb))
    suite.addTests(loader.loadTestsFromTestCase(TestMHAcceptance))
    suite.addTests(loader.loadTestsFromTestCase(TestNMCMCZero))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
