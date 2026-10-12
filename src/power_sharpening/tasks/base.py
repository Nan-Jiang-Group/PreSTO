
from abc import ABC, abstractmethod
from typing import Any, Dict


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    default_model = "qwen"
    uses_max_model_len = True
    use_chat_template = False

    @staticmethod
    def batch_to_problems(batch) -> list[Dict]:
        """Convert a DataLoader batch back into per-example dictionaries."""
        if isinstance(batch, (list, tuple)):
            return list(batch)
        if not isinstance(batch, dict):
            raise TypeError(f"Unsupported benchmark batch type: {type(batch)!r}")
        if not batch:
            return []

        first = next(iter(batch.values()))
        size = len(first)
        problems = []
        for index in range(size):
            problem = {}
            for key, values in batch.items():
                value = values[index]
                if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
                    value = value.item()
                problem[key] = value
            problems.append(problem)
        return problems


    def should_use_chat_template(self, model_name: str) -> bool:
        """Return whether this benchmark should render chat-formatted prompts.

        Default policy: wrap prompts in the model's chat template exactly when the selected model is a
        chat/instruct-tuned checkpoint; base (non-chat) models get raw prompts. Benchmarks whose prompt style is fixed
        regardless of model (e.g. HumanEval, LiveCodeBench) override this.
        """
        # Local import mirrors the concrete benchmarks and avoids a circular
        # import between this module and ``constants`` at load time.
        from .constants import model_uses_chat_template
        
        self.use_chat_template = model_uses_chat_template(model_name)
        return self.use_chat_template

    def evaluate_completions(
        self,
        problems: list[Dict],
        completions: list[str],
    ) -> list[Dict]:
        """Extract, grade, and format one completion per problem."""
        if len(completions) != len(problems):
            raise RuntimeError(
                f"Sampler returned {len(completions)} completions "
                f"for {len(problems)} prompts"
            )

        results = []
        for problem, response in zip(problems, completions):
            prediction = self.extract_completion(response, problem)
            is_correct, justification = self.check_correctness(
                problem, prediction
            )
            results.append(
                {
                    "completion": response,
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "justification": justification,
                }
            )
        return results

    def needs_extraction(
        self,
        parsed_answer: Any,
        completion: str | None = None,
        problem: Dict | None = None,
    ) -> bool:
        """Return whether a short greedy answer-extraction pass should run."""
        return False

    def extract_answer_text(
        self,
        completion: str,
        problem: Dict,
    ) -> str | None:
        """Cue appended to prompt + completion before greedy extraction."""
        return None

    def parse_extraction(self, extracted_text: str, problem: Dict) -> Any:
        """Parse the short greedy extraction continuation."""
        return self.extract_completion(extracted_text, problem)

    def extract_max_tokens(self) -> int:
        """Token budget for the optional greedy extraction continuation."""
        return 4

    def extract_stop_tokens(self) -> list[str]:
        """Stop tokens for the optional greedy extraction continuation."""
        return ["}", "\n"]

    def stop_tokens(self) -> list[str] | None:
        """Stop tokens for the main decode."""
        return None

    @abstractmethod
    def load_dataset(self):
        """Load the benchmark dataset."""
        pass


    @abstractmethod
    def format_prompt(self, problem: Dict, use_chat_template=False) -> str:
        """Format a problem into a prompt for the LLM."""
        pass

    @staticmethod
    @abstractmethod
    def extract_completion(response: str, problem: Dict) -> str:
        """Extract the completion from LLM response."""
        pass

    @staticmethod
    @abstractmethod
    def check_correctness(problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.
        Returns: (passed, result_message)
        """
        pass
