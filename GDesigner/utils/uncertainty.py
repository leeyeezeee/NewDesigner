import asyncio
import math
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar

import httpx
import numpy as np
from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from GDesigner.utils.ig_scorer import FinalAnswerScorer, TargetSpec


T = TypeVar("T")

_DEFAULT_JUDGE_TIMEOUT = 120.0
_DEFAULT_JUDGE_CONNECT_TIMEOUT = 10.0
_DEFAULT_JUDGE_MAX_RETRIES = 3
_DEFAULT_JUDGE_MAX_CONCURRENCY = 16
_DEFAULT_KLE_HEAT_T = 0.3

_NLI_LABELS = ("entailment", "neutral", "contradiction")
_NLI_LABEL_SCORES = {
    "entailment": 1.0,
    "neutral": 0.5,
    "contradiction": 0.0,
}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _judge_http_timeout(read_timeout: float, connect_timeout: float) -> httpx.Timeout:
    return httpx.Timeout(timeout=read_timeout, connect=connect_timeout)


def _parse_nli_label(verdict: str) -> str:
    normalized = re.sub(r"[^a-z]+", " ", verdict.lower()).strip()
    if not normalized:
        return "neutral"

    first_token = normalized.split()[0]
    if first_token.startswith("entail"):
        return "entailment"
    if first_token.startswith("neutral"):
        return "neutral"
    if first_token.startswith("contrad"):
        return "contradiction"

    matches = [
        label
        for label in _NLI_LABELS
        if re.search(rf"\b{label}\b", normalized)
    ]
    if len(matches) == 1:
        return matches[0]
    return "neutral"


def heat_kernel_language_entropy(
    weight_matrix: Iterable[Iterable[float]],
    heat_t: float = _DEFAULT_KLE_HEAT_T,
) -> Tuple[float, Dict[str, Any]]:
    if heat_t < 0:
        raise ValueError("KHEAT lengthscale --kle_heat_t must be non-negative.")

    weights = np.asarray(weight_matrix, dtype=float)
    if weights.size == 0:
        return 0.0, {
            "kernel": "heat",
            "kle_heat_t": float(heat_t),
            "weights": [],
            "laplacian_eigenvalues": [],
            "kernel_eigenvalues": [],
        }
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("KHEAT requires a square semantic weight matrix.")

    num_outputs = int(weights.shape[0])
    if num_outputs <= 1:
        return 0.0, {
            "kernel": "heat",
            "kle_heat_t": float(heat_t),
            "weights": weights.tolist(),
            "laplacian_eigenvalues": [],
            "kernel_eigenvalues": [1.0] if num_outputs == 1 else [],
        }

    weights = np.maximum((weights + weights.T) / 2.0, 0.0)
    np.fill_diagonal(weights, 0.0)
    laplacian = np.diag(weights.sum(axis=1)) - weights
    laplacian_eigenvalues = np.linalg.eigvalsh(laplacian)
    laplacian_eigenvalues = np.maximum(laplacian_eigenvalues, 0.0)
    kernel_eigenvalues = np.exp(-float(heat_t) * laplacian_eigenvalues)
    trace = float(kernel_eigenvalues.sum())
    if trace <= 0:
        return 0.0, {
            "kernel": "heat",
            "kle_heat_t": float(heat_t),
            "weights": weights.tolist(),
            "laplacian_eigenvalues": laplacian_eigenvalues.tolist(),
            "kernel_eigenvalues": [],
        }

    density_eigenvalues = kernel_eigenvalues / trace
    positive_eigenvalues = density_eigenvalues[density_eigenvalues > 0]
    entropy = -float(np.sum(positive_eigenvalues * np.log(positive_eigenvalues)))
    return entropy, {
        "kernel": "heat",
        "kle_heat_t": float(heat_t),
        "weights": weights.tolist(),
        "laplacian_eigenvalues": laplacian_eigenvalues.tolist(),
        "kernel_eigenvalues": density_eigenvalues.tolist(),
    }


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


def _semantic_judge_extra_body(model: str) -> Dict[str, Any]:
    """Qwen backends (e.g. vveai) require enable_thinking=false for non-streaming judge calls."""
    if "qwen" not in model.lower():
        return {}
    return {
        "enable_thinking": False,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }


class SemanticEntailmentJudge:
    def __init__(
        self,
        llm_name: Optional[str] = None,
        api_key: str = "",
        base_url: str = "",
        model_path: str = "",
        timeout: Optional[float] = None,
        connect_timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        max_concurrency: Optional[int] = None,
    ):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        self.timeout = (
            timeout
            if timeout is not None
            else _float_env("SEMANTIC_JUDGE_TIMEOUT", _DEFAULT_JUDGE_TIMEOUT)
        )
        self.connect_timeout = (
            connect_timeout
            if connect_timeout is not None
            else _float_env("SEMANTIC_JUDGE_CONNECT_TIMEOUT", _DEFAULT_JUDGE_CONNECT_TIMEOUT)
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else _int_env("SEMANTIC_JUDGE_MAX_RETRIES", _DEFAULT_JUDGE_MAX_RETRIES)
        )
        self.max_concurrency = max(
            1,
            max_concurrency
            if max_concurrency is not None
            else _int_env(
                "SEMANTIC_JUDGE_MAX_CONCURRENCY",
                _DEFAULT_JUDGE_MAX_CONCURRENCY,
            ),
        )
        self._request_semaphore = asyncio.Semaphore(self.max_concurrency)

        self.llm_name = (
            model_path
            or llm_name
            or os.getenv("SEMANTIC_JUDGE_MODEL")
            or "gpt-4o-mini"
        )
        self.api_key = (
            api_key
            or os.getenv("SEMANTIC_JUDGE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        self.base_url = (
            base_url
            or os.getenv("SEMANTIC_JUDGE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or ""
        )
        if self.base_url and not self.api_key:
            self.api_key = "EMPTY"
        self._client = None
        if self.llm_name and self.api_key:
            from openai import AsyncOpenAI

            client_kwargs = {
                "api_key": self.api_key,
                "timeout": _judge_http_timeout(self.timeout, self.connect_timeout),
                "max_retries": 0,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**client_kwargs)

    @property
    def is_configured(self) -> bool:
        return bool(self._client and self.llm_name)

    async def _create_completion(self, request_kwargs: Dict[str, Any]):
        async for attempt in AsyncRetrying(
            wait=wait_random_exponential(multiplier=1, max=60),
            stop=stop_after_attempt(max(1, self.max_retries)),
            retry=retry_if_exception_type(
                (APITimeoutError, APIConnectionError, RateLimitError)
            ),
            reraise=True,
        ):
            with attempt:
                async with self._request_semaphore:
                    return await self._client.chat.completions.create(**request_kwargs)

    async def nli_label(self, question: str, premise: str, hypothesis: str) -> str:
        if self._client is None:
            raise RuntimeError(
                "SemanticEntailmentJudge is not configured. For remote OpenAI, set "
                "--semantic_judge_llm_name and --semantic_judge_api_key or OPENAI_API_KEY. "
                "For local vLLM, set --semantic_judge_llm_name and "
                "--semantic_judge_base_url, for example http://localhost:8000/v1."
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
        request_kwargs: Dict[str, Any] = {
            "model": self.llm_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 32,
        }
        extra_body = _semantic_judge_extra_body(self.llm_name)
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        response = await self._create_completion(request_kwargs)
        verdict = response.choices[0].message.content or ""
        return _parse_nli_label(verdict)

    async def nli_score(self, question: str, premise: str, hypothesis: str) -> float:
        label = await self.nli_label(question, premise, hypothesis)
        return _NLI_LABEL_SCORES[label]

    async def entails(self, question: str, premise: str, hypothesis: str) -> bool:
        label = await self.nli_label(question, premise, hypothesis)
        return label == "entailment"

    async def semantic_weight_matrix(
        self,
        question: str,
        outputs: Iterable[Any],
    ) -> List[List[float]]:
        valid_outputs = [str(output).strip() for output in outputs if str(output).strip()]
        num_outputs = len(valid_outputs)
        weights = np.zeros((num_outputs, num_outputs), dtype=float)
        tasks = []
        task_pairs = []

        for i in range(num_outputs):
            for j in range(i + 1, num_outputs):
                if valid_outputs[i] == valid_outputs[j]:
                    weights[i, j] = weights[j, i] = 2.0
                    continue
                tasks.append(asyncio.gather(
                    self.nli_score(question, valid_outputs[i], valid_outputs[j]),
                    self.nli_score(question, valid_outputs[j], valid_outputs[i]),
                ))
                task_pairs.append((i, j))

        if tasks:
            pair_scores = await asyncio.gather(*tasks)
            for (i, j), (forward_score, backward_score) in zip(task_pairs, pair_scores):
                weights[i, j] = weights[j, i] = float(forward_score) + float(backward_score)

        return weights.tolist()

    async def equivalent(self, question: str, output_a: str, output_b: str) -> bool:
        if output_a.strip() == output_b.strip():
            return True
        forward, backward = await asyncio.gather(
            self.entails(question, output_a, output_b),
            self.entails(question, output_b, output_a),
            return_exceptions=True,
        )
        if isinstance(forward, Exception):
            raise forward
        if not forward:
            return False
        if isinstance(backward, Exception):
            raise backward
        return bool(backward)

    async def cluster_outputs(self, question: str, outputs: Iterable[Any]) -> List[str]:
        valid_outputs = [str(output) for output in outputs if str(output).strip()]
        clusters: List[List[str]] = []
        labels: List[str] = []
        for output in valid_outputs:
            label = ""
            if clusters:
                comparisons = await asyncio.gather(
                    *[
                        self.equivalent(question, output, cluster[0])
                        for cluster in clusters
                    ],
                    return_exceptions=True,
                )
            else:
                comparisons = []
            for cluster_idx, comparison in enumerate(comparisons):
                if isinstance(comparison, Exception):
                    raise comparison
                if comparison:
                    clusters[cluster_idx].append(output)
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
    heat_t: float = _DEFAULT_KLE_HEAT_T,
) -> Tuple[float, Dict[str, Any]]:
    valid_outputs = [str(output).strip() for output in outputs if str(output).strip()]
    if len(valid_outputs) <= 1:
        return 0.0, {
            "method": "semantic_entropy",
            "outputs": valid_outputs,
            "labels": ["cluster_0"] if valid_outputs else [],
        }

    # KLE is retained above for later re-enable; for now use cluster semantic entropy.
    labels = await judge.cluster_outputs(question, valid_outputs)
    entropy = semantic_entropy(labels)
    return entropy, {
        "method": "semantic_entropy",
        "outputs": valid_outputs,
        "labels": labels,
    }


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
    uncertainty_delta: float,
    negative_reward_scale: float,
    nonpositive_penalty: float,
) -> float:
    if uncertainty_delta > 0:
        return uncertainty_delta
    if uncertainty_delta < 0:
        return negative_reward_scale * uncertainty_delta
    return 0.0


def _normalize_edge_rewards(
    details: Dict[str, Dict[str, Any]],
    negative_reward_scale: float,
    nonpositive_penalty: float,
) -> Dict[str, float]:
    if not details:
        return {}

    uncertainty_deltas = [
        float(detail.get("uncertainty_delta", detail.get("entropy_delta", 0.0)))
        for detail in details.values()
    ]
    max_abs_delta = max(abs(delta) for delta in uncertainty_deltas)

    rewards: Dict[str, float] = {}
    for key, detail in details.items():
        uncertainty_delta = float(detail.get("uncertainty_delta", detail.get("entropy_delta", 0.0)))
        normalized_delta = (
            uncertainty_delta / max_abs_delta
            if max_abs_delta > 0
            else 0.0
        )
        reward = _edge_reward_from_delta(
            normalized_delta,
            negative_reward_scale=negative_reward_scale,
            nonpositive_penalty=nonpositive_penalty,
        )
        detail["normalized_uncertainty_delta"] = normalized_delta
        detail["normalized_entropy_delta"] = normalized_delta
        detail["reward"] = reward
        rewards[key] = reward
    return rewards


async def edge_entropy_rewards(
    graph,
    question: str,
    input_data: Any,
    judge: SemanticEntailmentJudge,
    num_entropy_samples: int,
    negative_reward_scale: float = 1.0,
    nonpositive_penalty: float = 0.01,
    kle_heat_t: float = _DEFAULT_KLE_HEAT_T,
    target_spec: Optional[TargetSpec] = None,
    ig_scorer: Optional[FinalAnswerScorer] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Measure each selected edge by removing it and comparing semantic uncertainty."""
    if not graph.edge_log_probs or num_entropy_samples <= 1:
        return {}, {}

    histories: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for node_id, node in graph.nodes.items():
        for history_item in node.execution_history:
            histories[(node_id, history_item["round"])] = history_item

    details: Dict[str, Dict[str, Any]] = {}
    after_cache: Dict[Tuple[str, int], Tuple[float, Dict[str, Any]]] = {}
    after_outputs_cache: Dict[Tuple[str, int], List[Any]] = {}
    after_score_cache: Dict[Tuple[str, int], Any] = {}
    scorer = ig_scorer or (FinalAnswerScorer() if target_spec is not None else None)

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
        if not before_outputs:
            continue

        after_cache_key = (target_id, round_idx)
        if after_cache_key in after_cache:
            before_uncertainty, before_uncertainty_details = await semantic_uncertainty(
                question,
                before_outputs,
                judge,
                heat_t=kle_heat_t,
            )
            after_uncertainty, after_uncertainty_details = after_cache[after_cache_key]
            after_outputs = after_outputs_cache.get(after_cache_key, [])
        else:
            after_outputs = history_item.get("entropy_samples", [])
            if not after_outputs:
                continue
            before_result, after_result = await asyncio.gather(
                semantic_uncertainty(question, before_outputs, judge, heat_t=kle_heat_t),
                semantic_uncertainty(question, after_outputs, judge, heat_t=kle_heat_t),
            )
            before_uncertainty, before_uncertainty_details = before_result
            after_uncertainty, after_uncertainty_details = after_result
            after_cache[after_cache_key] = (after_uncertainty, after_uncertainty_details)
            after_outputs_cache[after_cache_key] = list(after_outputs)
            history_item["entropy_samples"] = []

        uncertainty_delta = before_uncertainty - after_uncertainty
        ig_gain = None
        before_answer_score = None
        after_answer_score = None
        if target_spec is not None and scorer is not None:
            before_score_task = scorer.score_outputs(
                graph.decision_node,
                input_data,
                before_uncertainty_details.get("outputs", before_outputs),
                target_spec,
                cluster_labels=before_uncertainty_details.get("labels"),
            )
            if after_cache_key in after_score_cache:
                before_score = await before_score_task
                after_score = after_score_cache[after_cache_key]
            else:
                before_score, after_score = await asyncio.gather(
                    before_score_task,
                    scorer.score_outputs(
                        graph.decision_node,
                        input_data,
                        after_uncertainty_details.get("outputs", after_outputs),
                        target_spec,
                        cluster_labels=after_uncertainty_details.get("labels"),
                    ),
                )
                after_score_cache[after_cache_key] = after_score
            before_answer_score = before_score.score
            after_answer_score = after_score.score
            ig_gain = after_answer_score - before_answer_score
        details[key] = {
            "type": edge_type,
            "round": round_idx,
            "source": source_id,
            "target": target_id,
            "uncertainty_method": "semantic_entropy",
            "before_uncertainty": before_uncertainty,
            "after_uncertainty": after_uncertainty,
            "uncertainty_delta": uncertainty_delta,
            "normalized_uncertainty_delta": uncertainty_delta,
            "before_entropy": before_uncertainty,
            "after_entropy": after_uncertainty,
            "entropy_delta": uncertainty_delta,
            "normalized_entropy_delta": uncertainty_delta,
            "reward": 0.0,
            "before_uncertainty_details": before_uncertainty_details,
            "after_uncertainty_details": after_uncertainty_details,
        }
        if ig_gain is not None:
            details[key]["ig_gain"] = float(ig_gain)
            details[key]["before_answer_score"] = float(before_answer_score)
            details[key]["after_answer_score"] = float(after_answer_score)
            details[key]["before_answer_details"] = before_score.details
            details[key]["after_answer_details"] = after_score.details
            details[key]["ig_mode"] = target_spec.mode

    rewards = _normalize_edge_rewards(
        details,
        negative_reward_scale=negative_reward_scale,
        nonpositive_penalty=nonpositive_penalty,
    )
    return rewards, details

