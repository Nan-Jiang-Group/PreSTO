from typing import List, Dict, Any, Tuple
from .answer_extraction import answers_match


def exact_match(pred: str, gold: str) -> bool:
    return answers_match(pred, gold)


def compute_accuracy(
    predictions: List[str],
    gold_answers: List[str],
) -> float:
    if len(predictions) != len(gold_answers):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(gold_answers)} gold answers"
        )
        
    if len(predictions) == 0:
        return 0.0
        
    correct = sum([exact_match(pred, gold) for pred, gold in zip(predictions, gold_answers)])
    
    return correct / len(predictions)


def compute_accuracy_from_results(
    results: List[Dict[str, Any]],
    gold_answers: List[str],
    answer_key: str = "final_answer",
) -> Tuple[float, List[bool]]:
    predictions = [r.get(answer_key, "") for r in results]
    
    correct_list = [
        exact_match(pred, gold)
        for pred, gold in zip(predictions, gold_answers)
    ]
    
    accuracy = sum(correct_list) / len(correct_list) if correct_list else 0.0
    
    return accuracy, correct_list


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {}
        
    metrics = {
        "count": len(results),
        "avg_wall_time_s": sum(r.get("wall_time_s", 0) for r in results) / len(results),
        "total_wall_time_s": sum(r.get("wall_time_s", 0) for r in results),
        "avg_output_tokens": sum(r.get("output_tokens", 0) for r in results) / len(results),
        "total_output_tokens": sum(r.get("output_tokens", 0) for r in results),
    }
    if "internal_generated_tokens" in results[0]:
        metrics["avg_internal_tokens"] = sum(r.get("internal_generated_tokens", 0) for r in results)/ len(results)
        metrics["total_internal_tokens"] = sum(
            r.get("internal_generated_tokens", 0) for r in results
        )
        
    if "accept_rate" in results[0]:
        accept_rates = [r.get("accept_rate", 0) for r in results if r.get("num_steps", 0) > 0]
        if accept_rates:
            metrics["avg_accept_rate"] = sum(accept_rates) / len(accept_rates)
            
    if "num_steps" in results[0]:
        metrics["avg_mh_steps"] = sum(r.get("num_steps", 0) for r in results) / len(results)
        
    return metrics
