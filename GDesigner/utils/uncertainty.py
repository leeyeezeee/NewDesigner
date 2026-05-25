import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar


T = TypeVar("T")


def semantic_entropy(labels: Iterable[str]) -> float:
    valid_labels = [label for label in labels if label]
    if len(valid_labels) <= 1:
        return 0.0

    counts = Counter(valid_labels)
    total = len(valid_labels)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


class SemanticEntailmentJudge:
    def __init__(
        self,
        llm_name: Optional[str] = None,
        api_key: str = "",
        base_url: str = "",
        model_path: str = "",
    ):
        self.llm_name = llm_name or ""
        self.api_key = api_key
        self.base_url = base_url
        self.model_path = model_path
        self._llm = None
        if self.llm_name:
            if self.api_key:
                os.environ["API_KEY"] = self.api_key
            if self.base_url:
                os.environ["BASE_URL"] = self.base_url
            from GDesigner.llm.llm_registry import LLMRegistry
            self._llm = LLMRegistry.get(self.llm_name)
        self._cache: Dict[Tuple[str, str, str], bool] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self._llm or self.model_path)

    async def entails(self, question: str, premise: str, hypothesis: str) -> bool:
        key = (question, premise, hypothesis)
        if key in self._cache:
            return self._cache[key]
        if self._llm is None:
            raise RuntimeError(
                "SemanticEntailmentJudge is not configured. Pass --semantic_judge_llm_name "
                "or wire a local model through --semantic_judge_model_path."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict natural language inference judge. "
                    "Decide whether the premise entails the hypothesis for the given task. "
                    "Focus on the meaning of the answer and reasoning, not surface wording. "
                    "Return only one token: entailment, contradiction, or neutral."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{question}\n\n"
                    f"Premise:\n{premise}\n\n"
                    f"Hypothesis:\n{hypothesis}\n\n"
                    "Does the premise entail the hypothesis?"
                ),
            },
        ]
        response = await self._llm.agen(messages, temperature=0.0)
        if isinstance(response, list):
            response = response[0] if response else ""
        verdict = str(response).strip().lower()
        result = verdict.startswith("entail")
        self._cache[key] = result
        return result

    async def equivalent(self, question: str, output_a: str, output_b: str) -> bool:
        if output_a.strip() == output_b.strip():
            return True
        return await self.entails(question, output_a, output_b) and await self.entails(question, output_b, output_a)

    async def cluster_outputs(self, question: str, outputs: Iterable[Any]) -> List[str]:
        valid_outputs = [str(output) for output in outputs if str(output).strip()]
        clusters: List[List[str]] = []
        labels: List[str] = []
        for output in valid_outputs:
            label = ""
            for cluster_idx, cluster in enumerate(clusters):
                if await self.equivalent(question, output, cluster[0]):
                    cluster.append(output)
                    label = f"cluster_{cluster_idx}"
                    break
            if not label:
                clusters.append([output])
                label = f"cluster_{len(clusters) - 1}"
            labels.append(label)
        return labels


async def semantic_uncertainty(
    question: str,
    outputs: Iterable[T],
    judge: SemanticEntailmentJudge,
) -> Tuple[float, List[str]]:
    labels = await judge.cluster_outputs(question, outputs)
    return semantic_entropy(labels), labels


def edge_key(edge_info: Dict[str, Any]) -> str:
    return edge_info.get(
        "edge_key",
        f"{edge_info['type']}:{edge_info['round']}:{edge_info['source']}->{edge_info['target']}",
    )


def _flatten_outputs(results: Iterable[Any]) -> List[Any]:
    outputs = []
    for result in results:
        if isinstance(result, list):
            outputs.extend(result)
        else:
            outputs.append(result)
    return outputs


async def _sample_node_outputs(
    node,
    input_data: Any,
    spatial_info: Dict[str, Any],
    temporal_info: Dict[str, Any],
    num_samples: int,
) -> List[Any]:
    import asyncio

    tasks = [
        asyncio.create_task(node._async_execute(input_data, spatial_info, temporal_info))
        for _ in range(max(1, int(num_samples)))
    ]
    return _flatten_outputs(await asyncio.gather(*tasks, return_exceptions=False))


def _edge_reward_from_delta(
    entropy_delta: float,
    negative_reward_scale: float,
    nonpositive_penalty: float,
) -> float:
    if entropy_delta > 0:
        return entropy_delta
    return negative_reward_scale * entropy_delta - nonpositive_penalty


async def edge_entropy_rewards(
    graph,
    question: str,
    input_data: Any,
    judge: SemanticEntailmentJudge,
    num_entropy_samples: int,
    negative_reward_scale: float = 1.0,
    nonpositive_penalty: float = 0.01,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Measure each selected edge by removing only that edge from its target input."""
    if not graph.edge_log_probs or num_entropy_samples <= 1:
        return {}, {}

    histories: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for node_id, node in graph.nodes.items():
        for history_item in node.execution_history:
            histories[(node_id, history_item["round"])] = history_item

    rewards: Dict[str, float] = {}
    details: Dict[str, Dict[str, Any]] = {}
    after_cache: Dict[Tuple[str, int], Tuple[float, List[str]]] = {}

    for edge_info in graph.edge_log_probs:
        target_id = edge_info["target"]
        source_id = edge_info["source"]
        round_idx = edge_info["round"]
        edge_type = edge_info["type"]
        key = edge_key(edge_info)
        history_item = histories.get((target_id, round_idx))
        target_node = graph.nodes.get(target_id)
        if history_item is None or target_node is None:
            continue

        spatial_info = {
            node_id: dict(info)
            for node_id, info in history_item.get("spatial_info", {}).items()
        }
        temporal_info = {
            node_id: dict(info)
            for node_id, info in history_item.get("temporal_info", {}).items()
        }
        if edge_type == "spatial":
            if source_id not in spatial_info:
                continue
            before_spatial_info = dict(spatial_info)
            before_temporal_info = temporal_info
            before_spatial_info.pop(source_id, None)
        elif edge_type == "temporal":
            if source_id not in temporal_info:
                continue
            before_spatial_info = spatial_info
            before_temporal_info = dict(temporal_info)
            before_temporal_info.pop(source_id, None)
        else:
            continue

        before_outputs = await _sample_node_outputs(
            target_node,
            input_data,
            before_spatial_info,
            before_temporal_info,
            num_entropy_samples,
        )
        after_outputs = history_item.get("outputs", [])
        if not before_outputs or not after_outputs:
            continue

        before_entropy, before_labels = await semantic_uncertainty(question, before_outputs, judge)
        after_cache_key = (target_id, round_idx)
        if after_cache_key in after_cache:
            after_entropy, after_labels = after_cache[after_cache_key]
        else:
            after_entropy, after_labels = await semantic_uncertainty(question, after_outputs, judge)
            after_cache[after_cache_key] = (after_entropy, after_labels)

        entropy_delta = before_entropy - after_entropy
        reward = _edge_reward_from_delta(
            entropy_delta,
            negative_reward_scale=negative_reward_scale,
            nonpositive_penalty=nonpositive_penalty,
        )
        rewards[key] = reward
        details[key] = {
            "type": edge_type,
            "round": round_idx,
            "source": source_id,
            "target": target_id,
            "before_entropy": before_entropy,
            "after_entropy": after_entropy,
            "entropy_delta": entropy_delta,
            "reward": reward,
            "before_labels": before_labels,
            "after_labels": after_labels,
        }

    return rewards, details


def edge_semantic_loss(edge_log_probs, edge_rewards: dict, semantic_lambda: float):
    if semantic_lambda <= 0 or not edge_log_probs:
        return None

    losses = []
    for edge_info in edge_log_probs:
        reward = semantic_lambda * edge_rewards.get(edge_key(edge_info), 0.0)
        if reward != 0:
            # Negative rewards make gradient descent lower the probability of this selected edge.
            losses.append(-edge_info["log_prob"] * reward)
    return losses


def total_reward_with_edges(correctness_reward: float, edge_rewards: Dict[str, float], semantic_lambda: float) -> float:
    if semantic_lambda <= 0:
        return correctness_reward
    return correctness_reward + semantic_lambda * sum(edge_rewards.values())
