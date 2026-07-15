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
    """Qwen backends require thinking to be disabled through chat-template kwargs."""
    if "qwen" not in model.lower():
        return {}
    return {
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
) -> List[Any]:
    result = await node._async_execute(input_data, spatial_info, temporal_info)
    return _flatten_outputs([result])


def _require_nonblank_edge_outputs(
    outputs: Iterable[Any],
    *,
    key: str,
    round_idx: int,
    source_id: str,
    target_id: str,
    phase: str,
) -> List[Any]:
    output_list = list(outputs)
    if output_list and all(str(output).strip() for output in output_list):
        return output_list
    blank_positions = [
        index
        for index, output in enumerate(output_list)
        if not str(output).strip()
    ]
    raise RuntimeError(
        "Edge IG received missing or blank candidate output. "
        f"edge_key={key!r}, round={round_idx}, source={source_id!r}, "
        f"target={target_id!r}, phase={phase!r}, "
        f"output_count={len(output_list)}, blank_positions={blank_positions}"
    )


def _current_graph_output_info(graph) -> Dict[str, Dict[str, Any]]:
    output_info: Dict[str, Dict[str, Any]] = {}
    for node_id, node in graph.nodes.items():
        node_outputs = getattr(node, "outputs", [])
        if isinstance(node_outputs, list):
            if not node_outputs:
                continue
            node_output = node_outputs[-1]
        else:
            node_output = node_outputs
        output_info[node_id] = {
            "role": getattr(node, "role", ""),
            "output": node_output,
        }
    return output_info


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

    reward_key = (
        "ig_gain"
        if any("ig_gain" in detail for detail in details.values())
        else "uncertainty_delta"
    )
    reward_values = [
        float(detail.get(reward_key, detail.get("entropy_delta", 0.0)))
        for detail in details.values()
    ]
    max_abs_delta = max(abs(value) for value in reward_values)

    rewards: Dict[str, float] = {}
    for key, detail in details.items():
        reward_value = float(detail.get(reward_key, detail.get("entropy_delta", 0.0)))
        normalized_delta = (
            reward_value / max_abs_delta
            if max_abs_delta > 0
            else 0.0
        )
        reward = _edge_reward_from_delta(
            normalized_delta,
            negative_reward_scale=negative_reward_scale,
            nonpositive_penalty=nonpositive_penalty,
        )
        if reward_key == "ig_gain":
            detail["normalized_ig_gain"] = normalized_delta
        detail["normalized_uncertainty_delta"] = normalized_delta
        detail["normalized_entropy_delta"] = normalized_delta
        detail["reward"] = reward
        rewards[key] = reward
    return rewards


async def edge_entropy_rewards(
    graph,
    question: str,
    input_data: Any,
    judge: Optional[SemanticEntailmentJudge],
    num_entropy_samples: int,
    negative_reward_scale: float = 1.0,
    nonpositive_penalty: float = 0.01,
    kle_heat_t: float = _DEFAULT_KLE_HEAT_T,
    target_spec: Optional[TargetSpec] = None,
    ig_scorer: Optional[FinalAnswerScorer] = None,
    compute_rewards: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Measure each selected edge by removing it and scoring receiver outputs."""
    if not graph.edge_log_probs:
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

        try:
            before_outputs = await _sample_node_outputs(
                target_node,
                input_data,
                before_spatial_info,
                before_temporal_info,
            )
        except Exception as exc:
            raise RuntimeError(
                "Edge IG counterfactual generation failed. "
                f"edge_key={key!r}, round={round_idx}, source={source_id!r}, "
                f"target={target_id!r}, phase='before'"
            ) from exc
        before_outputs = _require_nonblank_edge_outputs(
            before_outputs,
            key=key,
            round_idx=round_idx,
            source_id=source_id,
            target_id=target_id,
            phase="before",
        )

        after_cache_key = (target_id, round_idx)
        if after_cache_key in after_outputs_cache:
            after_outputs = after_outputs_cache[after_cache_key]
        else:
            after_outputs = _require_nonblank_edge_outputs(
                history_item.get("entropy_samples", []),
                key=key,
                round_idx=round_idx,
                source_id=source_id,
                target_id=target_id,
                phase="after",
            )
            after_outputs_cache[after_cache_key] = list(after_outputs)

        if judge is None:
            before_uncertainty = 0.0
            after_uncertainty = 0.0
            before_uncertainty_details = {
                "method": "direct_final_answer_gain",
                "outputs": list(before_outputs),
                "labels": None,
            }
            after_uncertainty_details = {
                "method": "direct_final_answer_gain",
                "outputs": list(after_outputs),
                "labels": None,
            }
        elif after_cache_key in after_cache:
            before_uncertainty, before_uncertainty_details = await semantic_uncertainty(
                question,
                before_outputs,
                judge,
                heat_t=kle_heat_t,
            )
            after_uncertainty, after_uncertainty_details = after_cache[after_cache_key]
        else:
            before_result, after_result = await asyncio.gather(
                semantic_uncertainty(question, before_outputs, judge, heat_t=kle_heat_t),
                semantic_uncertainty(question, after_outputs, judge, heat_t=kle_heat_t),
            )
            before_uncertainty, before_uncertainty_details = before_result
            after_uncertainty, after_uncertainty_details = after_result
            after_cache[after_cache_key] = (after_uncertainty, after_uncertainty_details)

        uncertainty_delta = before_uncertainty - after_uncertainty
        ig_gain = None
        before_answer_score = None
        after_answer_score = None
        if target_spec is not None and scorer is not None:
            target_is_final = target_node is graph.decision_node
            if target_is_final and target_spec.mode != "execution":
                before_score_task = scorer.teacher_answer_logprob(
                    target_node,
                    input_data,
                    before_spatial_info,
                    before_temporal_info,
                    target_spec,
                )
            elif target_is_final:
                before_score_task = scorer.score_outputs(
                    target_node,
                    input_data,
                    before_outputs,
                    target_spec,
                    cluster_labels=None,
                )
            elif target_spec.mode != "execution":
                before_score_task = scorer.final_agent_teacher_answer_logprob(
                    graph.decision_node,
                    input_data,
                    before_outputs,
                    target_spec,
                    cluster_labels=None,
                    base_spatial_info=None,
                    candidate_id=target_id,
                    candidate_role=getattr(target_node, "role", "Candidate"),
                )
            else:
                before_score_task = scorer.final_agent_execution_score(
                    graph.decision_node,
                    input_data,
                    before_outputs,
                    target_spec,
                    cluster_labels=None,
                    base_spatial_info=None,
                    candidate_id=target_id,
                    candidate_role=getattr(target_node, "role", "Candidate"),
                )
            if after_cache_key in after_score_cache:
                before_score = await before_score_task
                after_score = after_score_cache[after_cache_key]
            else:
                if target_is_final and target_spec.mode != "execution":
                    after_score_task = scorer.teacher_answer_logprob(
                        target_node,
                        input_data,
                        spatial_info,
                        temporal_info,
                        target_spec,
                    )
                elif target_is_final:
                    after_score_task = scorer.score_outputs(
                        target_node,
                        input_data,
                        after_outputs,
                        target_spec,
                        cluster_labels=None,
                    )
                elif target_spec.mode != "execution":
                    after_score_task = scorer.final_agent_teacher_answer_logprob(
                        graph.decision_node,
                        input_data,
                        after_outputs,
                        target_spec,
                        cluster_labels=None,
                        base_spatial_info=None,
                        candidate_id=target_id,
                        candidate_role=getattr(target_node, "role", "Candidate"),
                    )
                else:
                    after_score_task = scorer.final_agent_execution_score(
                        graph.decision_node,
                        input_data,
                        after_outputs,
                        target_spec,
                        cluster_labels=None,
                        base_spatial_info=None,
                        candidate_id=target_id,
                        candidate_role=getattr(target_node, "role", "Candidate"),
                    )
                before_score, after_score = await asyncio.gather(
                    before_score_task,
                    after_score_task,
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
            "uncertainty_method": "semantic_entropy" if judge is not None else "direct_final_answer_gain",
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
            if target_spec.mode != "execution":
                details[key]["before_teacher_logprob"] = float(before_answer_score)
                details[key]["after_teacher_logprob"] = float(after_answer_score)
                details[key]["uncertainty_method"] = "final_agent_teacher_logprob_diff"
                details[key]["teacher_forcing_agent"] = "final_agent"
                details[key]["teacher_forcing_context"] = "local_candidate_output"
                details[key]["teacher_forcing_candidate_role"] = getattr(target_node, "role", "")
            else:
                details[key]["uncertainty_method"] = "final_agent_execution_score_diff"
                details[key]["scoring_agent"] = "final_agent"
                details[key]["scoring_context"] = "local_candidate_output"
                details[key]["execution_candidate_role"] = getattr(target_node, "role", "")
            details[key]["before_answer_details"] = before_score.details
            details[key]["after_answer_details"] = after_score.details
            details[key]["ig_mode"] = (
                "final_agent_teacher_logprob_diff"
                if target_spec.mode != "execution"
                else "final_agent_execution_score_diff"
            )

    rewards = (
        _normalize_edge_rewards(
            details,
            negative_reward_scale=negative_reward_scale,
            nonpositive_penalty=nonpositive_penalty,
        )
        if compute_rewards
        else {}
    )
    return rewards, details

