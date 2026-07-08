import ast
import asyncio
import inspect
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
        choices = ["A", "B", "C", "D"]
        return TargetSpec(dataset=dataset_key, mode="yesno_logprob", correct=_require_choice_target(correct, choices), choices=choices)
    if dataset_key == "aqua":
        choices = ["A", "B", "C", "D", "E"]
        return TargetSpec(dataset=dataset_key, mode="yesno_logprob", correct=_require_choice_target(correct, choices), choices=choices)
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


def _get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "", str(label)).upper()
    if normalized.startswith("YES"):
        return "YES"
    if normalized.startswith("NO"):
        return "NO"
    return normalized[:1]


def _choice_label_set(labels: Sequence[str]) -> set[str]:
    return {_normalize_label(label) for label in labels}


def _all_single_character_labels(labels: Sequence[str]) -> bool:
    return all(len(_normalize_label(label)) == 1 for label in labels)


def _standalone_label_from_text(text: Any, labels: Sequence[str]) -> Optional[str]:
    normalized_labels = _choice_label_set(labels)
    value = str(text).strip()
    if not value:
        return None

    if _all_single_character_labels(labels):
        matches = [
            match.group(1).upper()
            for match in re.finditer(
                r"(?<![A-Za-z0-9])([A-Za-z0-9])(?:[\)\].,:;])?(?![A-Za-z0-9])",
                value,
            )
            if match.group(1).upper() in normalized_labels
        ]
        unique_matches = sorted(set(matches))
        return unique_matches[0] if len(unique_matches) == 1 else None

    for label in labels:
        normalized = _normalize_label(label)
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(str(label))}(?![A-Za-z0-9])", value, flags=re.IGNORECASE):
            return normalized
    return None


def _require_choice_target(correct: Any, labels: Sequence[str]) -> str:
    target = _standalone_label_from_text(correct, labels)
    if target is None:
        label_text = ", ".join(str(label) for label in labels)
        raise ValueError(
            f"Could not extract a unique correct option label from {correct!r}; "
            f"expected one of: {label_text}."
        )
    return target


def _completion_label_from_token(token: str, labels: Sequence[str]) -> Optional[str]:
    normalized_labels = _choice_label_set(labels)
    value = re.sub(r"^[^A-Za-z0-9]+", "", str(token).strip())
    if not value:
        return None

    if _all_single_character_labels(labels):
        match = re.match(r"^([A-Za-z0-9])(?:[\)\].,:;]|$)", value)
        if match:
            normalized = match.group(1).upper()
            return normalized if normalized in normalized_labels else None
        return None

    for label in labels:
        label_text = str(label).strip()
        normalized = _normalize_label(label_text)
        if re.match(rf"^{re.escape(label_text)}(?:[^A-Za-z0-9]|$)", value, flags=re.IGNORECASE):
            return normalized if normalized in normalized_labels else None
    return None


def _extract_python_code(output: Any) -> str:
    text = str(output)
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _humaneval_entry_point(test_source: str) -> Optional[str]:
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "check":
            continue
        if node.args and isinstance(node.args[0], ast.Name):
            return node.args[0].id
    return None


def _humaneval_assert_tests(test_source: str) -> List[str]:
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []

    check_fn = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "check"
        ),
        None,
    )
    if check_fn is None:
        return []

    return [
        ast.unparse(statement)
        for statement in check_fn.body
        if isinstance(statement, ast.Assert)
    ]


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

    async def teacher_answer_logprob(
        self,
        node,
        input_data: Dict[str, Any],
        spatial_info: Dict[str, Any],
        temporal_info: Dict[str, Any],
        target_spec: TargetSpec,
    ) -> ScoreResult:
        target_answer = self._teacher_answer_text(target_spec)
        if not target_answer:
            return ScoreResult(
                score=float(_LOGPROB_FLOOR),
                mode="teacher_logprob",
                details={"target_answer": target_answer, "error": "empty target answer"},
            )

        processed = node._process_inputs(input_data, spatial_info, temporal_info)
        if inspect.isawaitable(processed):
            processed = await processed
        system_prompt, user_prompt = processed
        messages = self._teacher_messages(system_prompt, user_prompt)

        try:
            score, details = await self._completion_target_logprob(
                node.llm,
                messages,
                target_answer,
            )
        except Exception as exc:
            score, details = await self._generated_target_logprob(
                node.llm,
                messages,
                target_answer,
                str(exc),
            )

        details["target_answer"] = target_answer
        return ScoreResult(
            score=float(score),
            mode="teacher_logprob",
            details=details,
        )

    async def final_agent_teacher_answer_logprob(
        self,
        decision_node,
        input_data: Dict[str, Any],
        outputs: Iterable[Any],
        target_spec: TargetSpec,
        cluster_labels: Optional[Sequence[str]] = None,
        base_spatial_info: Optional[Dict[str, Any]] = None,
        candidate_id: str = "candidate",
        candidate_role: str = "Candidate",
    ) -> ScoreResult:
        output_list, labels = _valid_outputs_and_labels(outputs, cluster_labels)
        if not output_list:
            return ScoreResult(
                score=float(_LOGPROB_FLOOR),
                mode="final_agent_teacher_logprob",
                details={"num_outputs": 0, "error": "empty candidate outputs"},
            )

        representatives, aggregation = _cluster_representatives(output_list, labels)
        scores = await asyncio.gather(*[
            self._final_agent_single_output_teacher_logprob(
                decision_node,
                input_data,
                representative.output,
                target_spec,
                base_spatial_info=base_spatial_info,
                candidate_id=candidate_id,
                candidate_role=candidate_role,
            )
            for representative in representatives
        ])
        weighted_score = sum(
            representative.weight * float(score.score)
            for representative, score in zip(representatives, scores)
        )
        return ScoreResult(
            score=float(weighted_score),
            mode="final_agent_teacher_logprob",
            details={
                "aggregation": aggregation,
                "num_outputs": len(output_list),
                "num_clusters": len(representatives),
                "teacher_forcing_agent": "final_agent",
                "clusters": [
                    {
                        "label": representative.label,
                        "count": representative.count,
                        "weight": representative.weight,
                        "score": float(score.score),
                        "details": score.details,
                    }
                    for representative, score in zip(representatives, scores)
                ],
            },
        )

    async def _final_agent_single_output_teacher_logprob(
        self,
        decision_node,
        input_data: Dict[str, Any],
        output: Any,
        target_spec: TargetSpec,
        base_spatial_info: Optional[Dict[str, Any]] = None,
        candidate_id: str = "candidate",
        candidate_role: str = "Candidate",
    ) -> ScoreResult:
        spatial_info = {
            node_id: dict(info)
            for node_id, info in (base_spatial_info or {}).items()
        }
        spatial_info[candidate_id] = {
            "role": candidate_role,
            "output": str(output),
        }
        result = await self.teacher_answer_logprob(
            decision_node,
            input_data,
            spatial_info,
            {},
            target_spec,
        )
        result.details["teacher_forcing_agent"] = "final_agent"
        result.details["candidate_id"] = candidate_id
        result.details["candidate_role"] = candidate_role
        result.details["candidate_output"] = str(output)
        return result

    async def final_agent_execution_score(
        self,
        decision_node,
        input_data: Dict[str, Any],
        outputs: Iterable[Any],
        target_spec: TargetSpec,
        cluster_labels: Optional[Sequence[str]] = None,
        base_spatial_info: Optional[Dict[str, Any]] = None,
        candidate_id: str = "candidate",
        candidate_role: str = "Candidate",
    ) -> ScoreResult:
        output_list, labels = _valid_outputs_and_labels(outputs, cluster_labels)
        if not output_list:
            return ScoreResult(
                score=0.0,
                mode="final_agent_execution",
                details={"num_outputs": 0, "error": "empty candidate outputs"},
            )

        representatives, aggregation = _cluster_representatives(output_list, labels)
        scores = await asyncio.gather(*[
            self._final_agent_single_output_execution_score(
                decision_node,
                input_data,
                representative.output,
                target_spec,
                base_spatial_info=base_spatial_info,
                candidate_id=candidate_id,
                candidate_role=candidate_role,
            )
            for representative in representatives
        ])
        weighted_score = sum(
            representative.weight * float(score.score)
            for representative, score in zip(representatives, scores)
        )
        return ScoreResult(
            score=float(weighted_score),
            mode="final_agent_execution",
            details={
                "aggregation": aggregation,
                "num_outputs": len(output_list),
                "num_clusters": len(representatives),
                "scoring_agent": "final_agent",
                "clusters": [
                    {
                        "label": representative.label,
                        "count": representative.count,
                        "weight": representative.weight,
                        "score": float(score.score),
                        "details": score.details,
                    }
                    for representative, score in zip(representatives, scores)
                ],
            },
        )

    async def _final_agent_single_output_execution_score(
        self,
        decision_node,
        input_data: Dict[str, Any],
        output: Any,
        target_spec: TargetSpec,
        base_spatial_info: Optional[Dict[str, Any]] = None,
        candidate_id: str = "candidate",
        candidate_role: str = "Candidate",
    ) -> ScoreResult:
        spatial_info = {
            node_id: dict(info)
            for node_id, info in (base_spatial_info or {}).items()
        }
        spatial_info[candidate_id] = {
            "role": candidate_role,
            "output": str(output),
        }
        generated = decision_node._async_execute(input_data, spatial_info, {})
        if inspect.isawaitable(generated):
            generated = await generated
        score = await asyncio.to_thread(self._execution_score, generated, target_spec)
        return ScoreResult(
            score=float(score),
            mode="final_agent_execution",
            details={
                "scoring_agent": "final_agent",
                "candidate_id": candidate_id,
                "candidate_role": candidate_role,
                "candidate_output": str(output),
                "final_agent_output": str(generated),
            },
        )

    def _teacher_answer_text(self, target_spec: TargetSpec) -> str:
        return str(target_spec.correct).strip()

    def _teacher_messages(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> List[Dict[str, str]]:
        user_prompt = (
            f"{user_prompt}\n\n"
            "Return only the final answer. Do not include reasoning, explanation, "
            "units unless required by the answer, or any extra text."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _execution_score(self, output: Any, target_spec: TargetSpec) -> float:
        from GDesigner.tools.coding.python_executor import PyExecutor

        tests = list(target_spec.tests or [])
        if not tests:
            raise ValueError("HumanEval IG scoring requires execution tests.")
        code = _extract_python_code(output)
        if target_spec.dataset == "humaneval":
            return self._humaneval_execution_score(code, tests)
        is_solved, _, _ = PyExecutor().execute(code, tests, timeout=100, verbose=False)
        return 1.0 if is_solved else 0.0

    def _humaneval_execution_score(self, code: str, tests: Sequence[str]) -> float:
        from GDesigner.tools.coding.python_executor import PyExecutor

        per_assert_tests: List[str] = []
        for test_source in tests:
            entry_point = _humaneval_entry_point(test_source)
            assert_tests = _humaneval_assert_tests(test_source)
            if entry_point is None:
                raise ValueError("HumanEval IG scoring could not parse the test entry point.")
            if not assert_tests:
                raise ValueError("HumanEval IG scoring requires at least one assert test.")
            per_assert_tests.extend([
                f"candidate = {entry_point}\n{assert_test}"
                for assert_test in assert_tests
            ])

        if not per_assert_tests:
            raise ValueError("HumanEval IG scoring requires at least one assert test.")

        _, _, states = PyExecutor().execute(code, per_assert_tests, timeout=100, verbose=False)
        if len(states) != len(per_assert_tests):
            raise RuntimeError("HumanEval IG scoring received an incomplete assertion result set.")
        return sum(1.0 for state in states if state) / len(per_assert_tests)

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
            messages = self._verifier_messages(decision_node, input_data, output, target_spec)
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
            f"Answer with exactly one option label from: {label_text}.\n"
            "Your entire reply must be exactly one label token and nothing else."
        )
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _verifier_messages(
        self,
        decision_node,
        input_data: Dict[str, Any],
        output: Any,
        target_spec: TargetSpec,
    ) -> List[Dict[str, str]]:
        role = decision_node.prompt_set.get_decision_role()
        task = input_data.get("task", str(input_data))
        system_prompt = (
            f"{role}\n"
            "You are a strict answer verifier. Reply with exactly one token: Yes or No."
        )
        if target_spec.choices:
            label_text = ", ".join(str(label) for label in target_spec.choices)
            user_prompt = (
                f"Task:\n{task}\n\n"
                f"Candidate response:\n{output}\n\n"
                f"Valid option labels:\n{label_text}\n\n"
                f"Reference correct option label:\n{target_spec.correct}\n\n"
                "Does the candidate response choose or clearly imply the reference correct option label? "
                "Reply with Yes or No only."
            )
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        user_prompt = (
            f"Task:\n{task}\n\n"
            f"Candidate response:\n{output}\n\n"
            f"Reference final answer:\n{target_spec.correct}\n\n"
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
            normalized = _completion_label_from_token(token, labels)
            if normalized in label_logprobs:
                label_logprobs[normalized] = max(label_logprobs[normalized], logprob)

        normalized_target = _normalize_label(target)
        if normalized_target not in label_logprobs:
            label_text = ", ".join(str(label) for label in labels)
            raise ValueError(
                f"Target label {target!r} is not one of the scoring labels: {label_text}."
            )
        target_logprob = label_logprobs.get(normalized_target, _LOGPROB_FLOOR)
        return target_logprob - _logsumexp(list(label_logprobs.values()))

    def _completion_prompt_prefix(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    async def _completion_target_logprob(
        self,
        llm,
        messages: List[Dict[str, str]],
        target_answer: str,
    ) -> tuple[float, Dict[str, Any]]:
        from openai import AsyncOpenAI

        from GDesigner.llm.gpt_chat import (
            _agent_base_url,
            _is_openai_compatible,
            _openai_client_kwargs,
        )

        base_url = _agent_base_url()
        if not _is_openai_compatible(base_url):
            raise RuntimeError("Teacher-forcing IG requires an OpenAI-compatible agent backend.")

        prompt_prefix = self._completion_prompt_prefix(messages)
        prompt = prompt_prefix + target_answer
        response = await AsyncOpenAI(**_openai_client_kwargs(base_url)).completions.create(
            model=llm.model_name,
            prompt=prompt,
            max_tokens=0,
            temperature=0.0,
            logprobs=1,
            echo=True,
        )

        choice = response.choices[0]
        logprobs = _get_attr_or_key(choice, "logprobs")
        tokens = list(_get_attr_or_key(logprobs, "tokens", []) or [])
        token_logprobs = list(_get_attr_or_key(logprobs, "token_logprobs", []) or [])
        text_offsets = list(_get_attr_or_key(logprobs, "text_offset", []) or [])
        if not tokens or not token_logprobs or not text_offsets:
            raise RuntimeError("Completion echo response did not include prompt token logprobs.")

        target_start = len(prompt_prefix)
        target_end = len(prompt)
        target_logprobs = []
        target_tokens = []
        for idx, (token, logprob, offset) in enumerate(zip(tokens, token_logprobs, text_offsets)):
            if logprob is None:
                continue
            next_offset = text_offsets[idx + 1] if idx + 1 < len(text_offsets) else len(prompt)
            if next_offset <= target_start or int(offset) >= target_end:
                continue
            target_logprobs.append(float(logprob))
            target_tokens.append(str(token))

        if not target_logprobs:
            raise RuntimeError("No target-answer token logprobs were found in completion echo response.")

        return sum(target_logprobs) / len(target_logprobs), {
            "method": "completion_echo_teacher_logprob",
            "num_target_tokens": len(target_logprobs),
            "target_tokens": target_tokens,
        }

    async def _generated_target_logprob(
        self,
        llm,
        messages: List[Dict[str, str]],
        target_answer: str,
        fallback_reason: str,
    ) -> tuple[float, Dict[str, Any]]:
        max_tokens = max(4, len(str(target_answer).split()) + 8)
        generation = await llm.agen(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            num_comps=1,
            return_logprobs=True,
        )
        content = generation.content if hasattr(generation, "content") else str(generation)
        token_logprobs = list(getattr(generation, "token_logprobs", []) or [])
        logprobs = [
            float(item.logprob)
            for item in token_logprobs
            if getattr(item, "logprob", None) is not None
        ]
        score = sum(logprobs) / len(logprobs) if logprobs else float(_LOGPROB_FLOOR)
        return score, {
            "method": "generated_answer_logprob_fallback",
            "fallback_reason": fallback_reason,
            "generated_answer": content,
            "target_match": self._answers_match(content, target_answer),
            "num_generated_tokens": len(logprobs),
        }

    def _answers_match(self, generated: Any, target: Any) -> bool:
        return str(generated).strip().lower() == str(target).strip().lower()
