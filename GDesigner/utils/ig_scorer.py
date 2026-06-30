import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

_LOGPROB_FLOOR = -20.0


@dataclass
class TargetSpec:
    dataset: str
    mode: str
    correct: str = ""
    choices: Optional[List[str]] = None
    tests: Optional[List[str]] = None


@dataclass
class ScoreResult:
    score: float
    mode: str
    details: Dict[str, Any]


@dataclass
class ClusterRepresentative:
    label: str
    output: Any
    count: int
    weight: float


def make_target_spec(
    dataset: str,
    correct: str = "",
    *,
    tests: Optional[Iterable[str]] = None,
) -> TargetSpec:
    dataset_key = dataset.lower()
    if dataset_key == "mmlu":
        return TargetSpec(dataset=dataset_key, mode="choice_logprob", correct=str(correct), choices=["A", "B", "C", "D"])
    if dataset_key == "aqua":
        return TargetSpec(dataset=dataset_key, mode="choice_logprob", correct=str(correct), choices=["A", "B", "C", "D", "E"])
    if dataset_key == "humaneval":
        return TargetSpec(dataset=dataset_key, mode="execution", tests=list(tests or []))
    return TargetSpec(dataset=dataset_key, mode="yesno_logprob", correct=str(correct), choices=["Yes", "No"])


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    max_value = max(values)
    if not math.isfinite(max_value):
        return max_value
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "", str(label)).upper()
    if normalized.startswith("YES"):
        return "YES"
    if normalized.startswith("NO"):
        return "NO"
    return normalized[:1]


def _extract_python_code(output: Any) -> str:
    text = str(output)
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _completion_token_logprob(logprob_item: Any) -> tuple[str, float]:
    if isinstance(logprob_item, dict):
        return str(logprob_item.get("token", "")), float(logprob_item.get("logprob", _LOGPROB_FLOOR))
    return str(getattr(logprob_item, "token", "")), float(getattr(logprob_item, "logprob", _LOGPROB_FLOOR))


def _top_logprobs_from_response(response: Any) -> List[Any]:
    choice = response.choices[0]
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None and isinstance(choice, dict):
        logprobs = choice.get("logprobs")
    content = getattr(logprobs, "content", None) if logprobs is not None else None
    if content is None and isinstance(logprobs, dict):
        content = logprobs.get("content")
    if not content:
        return []
    first_token = content[0]
    top_logprobs = getattr(first_token, "top_logprobs", None)
    if top_logprobs is None and isinstance(first_token, dict):
        top_logprobs = first_token.get("top_logprobs")
    return list(top_logprobs or [])


def _merge_extra_body(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _qwen_logprob_extra_body(model: str) -> Dict[str, Any]:
    if "qwen" not in model.lower():
        return {}
    return {
        "enable_thinking": False,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }


def _valid_outputs_and_labels(
    outputs: Iterable[Any],
    cluster_labels: Optional[Sequence[str]],
) -> tuple[List[Any], Optional[List[str]]]:
    raw_outputs = list(outputs)
    if cluster_labels is None:
        return [output for output in raw_outputs if str(output).strip()], None

    labels = list(cluster_labels)
    valid_outputs = [output for output in raw_outputs if str(output).strip()]
    if len(labels) == len(raw_outputs):
        pairs = [
            (output, label)
            for output, label in zip(raw_outputs, labels)
            if str(output).strip() and str(label).strip()
        ]
    elif len(labels) == len(valid_outputs):
        pairs = [
            (output, label)
            for output, label in zip(valid_outputs, labels)
            if str(label).strip()
        ]
    else:
        raise ValueError(
            "cluster_labels must align with outputs or with the non-empty outputs."
        )

    if not pairs:
        return [], []
    return [output for output, _ in pairs], [str(label) for _, label in pairs]


def _cluster_representatives(
    outputs: Sequence[Any],
    labels: Optional[Sequence[str]],
) -> tuple[List[ClusterRepresentative], str]:
    if not labels:
        total = len(outputs)
        return [
            ClusterRepresentative(
                label=f"sample_{idx}",
                output=output,
                count=1,
                weight=1.0 / total,
            )
            for idx, output in enumerate(outputs)
        ], "mean"

    cluster_order: List[str] = []
    clusters: Dict[str, Dict[str, Any]] = {}
    for output, label in zip(outputs, labels):
        if label not in clusters:
            cluster_order.append(label)
            clusters[label] = {"output": output, "count": 0}
        clusters[label]["count"] += 1

    total = float(sum(cluster["count"] for cluster in clusters.values()))
    representatives = [
        ClusterRepresentative(
            label=label,
            output=clusters[label]["output"],
            count=int(clusters[label]["count"]),
            weight=float(clusters[label]["count"] / total),
        )
        for label in cluster_order
    ]
    return representatives, "cluster_weighted"


class FinalAnswerScorer:
    def __init__(self, top_logprobs: int = 20):
        self.top_logprobs = max(1, int(top_logprobs))

    async def score_outputs(
        self,
        decision_node,
        input_data: Dict[str, Any],
        outputs: Iterable[Any],
        target_spec: TargetSpec,
        cluster_labels: Optional[Sequence[str]] = None,
    ) -> ScoreResult:
        output_list, labels = _valid_outputs_and_labels(outputs, cluster_labels)
        if not output_list:
            return ScoreResult(0.0, target_spec.mode, {"num_outputs": 0})

        representatives, aggregation = _cluster_representatives(output_list, labels)
        if target_spec.mode == "execution":
            scores = await asyncio.gather(*[
                asyncio.to_thread(self._execution_score, representative.output, target_spec)
                for representative in representatives
            ])
        else:
            scores = await asyncio.gather(*[
                self._logprob_score(decision_node, input_data, representative.output, target_spec)
                for representative in representatives
            ])

        weighted_score = sum(
            representative.weight * float(score)
            for representative, score in zip(representatives, scores)
        )
        return ScoreResult(
            score=float(weighted_score),
            mode=target_spec.mode,
            details={
                "aggregation": aggregation,
                "num_outputs": len(output_list),
                "num_clusters": len(representatives),
                "clusters": [
                    {
                        "label": representative.label,
                        "count": representative.count,
                        "weight": representative.weight,
                        "score": float(score),
                    }
                    for representative, score in zip(representatives, scores)
                ],
            },
        )

    def _execution_score(self, output: Any, target_spec: TargetSpec) -> float:
        from GDesigner.tools.coding.python_executor import PyExecutor

        tests = list(target_spec.tests or [])
        if not tests:
            raise ValueError("HumanEval IG scoring requires execution tests.")
        code = _extract_python_code(output)
        is_solved, _, _ = PyExecutor().execute(code, tests, timeout=100, verbose=False)
        return 1.0 if is_solved else 0.0

    async def _logprob_score(
        self,
        decision_node,
        input_data: Dict[str, Any],
        output: Any,
        target_spec: TargetSpec,
    ) -> float:
        if target_spec.mode == "choice_logprob":
            labels = list(target_spec.choices or [])
            target = target_spec.correct
            messages = self._decision_messages(decision_node, input_data, output, labels)
        elif target_spec.mode == "yesno_logprob":
            labels = ["Yes", "No"]
            target = "Yes"
            messages = self._verifier_messages(decision_node, input_data, output, target_spec.correct)
        else:
            return 0.0
        return await self._conditional_label_logprob(decision_node.llm, messages, labels, target)

    def _decision_messages(
        self,
        decision_node,
        input_data: Dict[str, Any],
        output: Any,
        labels: Sequence[str],
    ) -> List[Dict[str, str]]:
        spatial_info = {
            "candidate": {
                "role": "Candidate",
                "output": str(output),
            }
        }
        system_prompt, user_prompt = decision_node._process_inputs(input_data, spatial_info, {})
        label_text = ", ".join(str(label) for label in labels)
        user_prompt = (
            f"{user_prompt}\n\n"
            f"Answer with exactly one option label from: {label_text}."
        )
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _verifier_messages(
        self,
        decision_node,
        input_data: Dict[str, Any],
        output: Any,
        correct_answer: str,
    ) -> List[Dict[str, str]]:
        role = decision_node.prompt_set.get_decision_role()
        task = input_data.get("task", str(input_data))
        system_prompt = (
            f"{role}\n"
            "You are a strict answer verifier. Reply with exactly one token: Yes or No."
        )
        user_prompt = (
            f"Task:\n{task}\n\n"
            f"Candidate response:\n{output}\n\n"
            f"Reference final answer:\n{correct_answer}\n\n"
            "Does the candidate response imply the same final answer as the reference? "
            "Reply with Yes or No only."
        )
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    async def _conditional_label_logprob(
        self,
        llm,
        messages: List[Dict[str, str]],
        labels: Sequence[str],
        target: str,
    ) -> float:
        from openai import AsyncOpenAI

        from GDesigner.llm.gpt_chat import (
            _agent_base_url,
            _chat_completion_extra_body,
            _is_openai_compatible,
            _openai_client_kwargs,
        )

        base_url = _agent_base_url()
        if not _is_openai_compatible(base_url):
            raise RuntimeError("IG logprob scoring requires an OpenAI-compatible agent backend.")

        request_kwargs = {
            "model": llm.model_name,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "n": 1,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }
        extra_body = _merge_extra_body(
            _chat_completion_extra_body(llm.model_name),
            _qwen_logprob_extra_body(llm.model_name),
        )
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        response = await AsyncOpenAI(**_openai_client_kwargs(base_url)).chat.completions.create(**request_kwargs)

        label_logprobs = {_normalize_label(label): _LOGPROB_FLOOR for label in labels}
        for logprob_item in _top_logprobs_from_response(response):
            token, logprob = _completion_token_logprob(logprob_item)
            normalized = _normalize_label(token)
            if normalized in label_logprobs:
                label_logprobs[normalized] = max(label_logprobs[normalized], logprob)

        normalized_target = _normalize_label(target)
        target_logprob = label_logprobs.get(normalized_target, _LOGPROB_FLOOR)
        return target_logprob - _logsumexp(list(label_logprobs.values()))
