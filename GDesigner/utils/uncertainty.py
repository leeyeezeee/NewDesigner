import math
from collections import Counter
from typing import Callable, Iterable, List, Tuple, TypeVar


T = TypeVar("T")


def normalized_entropy(labels: Iterable[str]) -> float:
    valid_labels = [label for label in labels if label]
    if len(valid_labels) <= 1:
        return 0.0

    counts = Counter(valid_labels)
    if len(counts) <= 1:
        return 0.0

    total = len(valid_labels)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(counts))


def answer_uncertainty(
    outputs: Iterable[T],
    label_fn: Callable[[T], str],
) -> Tuple[float, List[str]]:
    labels = []
    for output in outputs:
        try:
            labels.append(label_fn(output))
        except Exception:
            labels.append("")
    return normalized_entropy(labels), labels


def uncertainty_adjusted_utility(
    correctness_reward: float,
    uncertainty: float,
    uncertainty_lambda: float,
) -> float:
    if uncertainty_lambda <= 0:
        return correctness_reward
    uncertainty = min(max(uncertainty, 0.0), 1.0)
    return correctness_reward * (1.0 + uncertainty_lambda * (1.0 - uncertainty))
