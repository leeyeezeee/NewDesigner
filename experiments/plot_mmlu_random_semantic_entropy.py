import argparse
import asyncio
import csv
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from GDesigner.graph.graph import Graph
from GDesigner.tools.coding.python_executor import PyExecutor
from GDesigner.tools.reader.readers import JSONLReader, JSONReader
from datasets.MMLU.download import download as download_mmlu
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from datasets.gsm8k_dataset import (
    gsm_data_process,
    gsm_get_predict,
    multiarith_data_process,
    svamp_data_process,
)
from datasets.mmlu_dataset import MMLUDataset
from GDesigner.utils.uncertainty import (
    SemanticEntailmentJudge,
    semantic_entropy,
    semantic_uncertainty,
)


RecordToInput = Callable[[Any], Dict[str, Any]]
RecordToTarget = Callable[[Any], str]
PredictionParser = Callable[[Any], str]
CorrectnessFn = Callable[[str, str], bool]


@dataclass(frozen=True)
class DatasetDefaults:
    dataset_json: Optional[str]
    graph_domain: str
    agent_names: List[str]
    agent_nums: List[int]
    num_rounds: int
    decision_method: str


@dataclass
class DatasetBundle:
    name: str
    records: Sequence[Any]
    graph_domain: str
    agent_names: List[str]
    agent_nums: List[int]
    num_rounds: int
    decision_method: str
    record_to_input: RecordToInput
    record_to_target: RecordToTarget
    parse_prediction: PredictionParser
    is_correct: CorrectnessFn


@dataclass
class GraphSampleResult:
    raw_final_answer: str
    predicted_answer: str


@dataclass
class InferenceResult:
    dataset: str
    index: int
    input_dict: Dict[str, Any]
    raw_final_answers: List[str]
    predicted_answers: List[str]
    target_answer: str
    modal_predicted_answer: str
    is_correct: bool


@dataclass(frozen=True)
class AnswerSpan:
    text: str
    start: int
    end: int


DATASET_DEFAULTS: Dict[str, DatasetDefaults] = {
    "mmlu": DatasetDefaults(
        dataset_json=None,
        graph_domain="mmlu",
        agent_names=["AnalyzeAgent"],
        agent_nums=[6],
        num_rounds=1,
        decision_method="FinalRefer",
    ),
    "gsm8k": DatasetDefaults(
        dataset_json="datasets/gsm8k/gsm8k.jsonl",
        graph_domain="gsm8k",
        agent_names=["MathSolver"],
        agent_nums=[5],
        num_rounds=1,
        decision_method="FinalRefer",
    ),
    "aqua": DatasetDefaults(
        dataset_json="datasets/AQuA/AQuA.jsonl",
        graph_domain="aqua",
        agent_names=["MathSolver_aqua"],
        agent_nums=[5],
        num_rounds=1,
        decision_method="FinalRefer",
    ),
    "svamp": DatasetDefaults(
        dataset_json="datasets/SVAMP/SVAMP.json",
        graph_domain="gsm8k",
        agent_names=["MathSolver"],
        agent_nums=[5],
        num_rounds=1,
        decision_method="FinalRefer",
    ),
    "multiarith": DatasetDefaults(
        dataset_json="datasets/MultiArith/MultiArith.json",
        graph_domain="gsm8k",
        agent_names=["MathSolver"],
        agent_nums=[5],
        num_rounds=1,
        decision_method="FinalRefer",
    ),
    "humaneval": DatasetDefaults(
        dataset_json="datasets/humaneval/humaneval-py.jsonl",
        graph_domain="humaneval",
        agent_names=["CodeWriting"],
        agent_nums=[5],
        num_rounds=2,
        decision_method="FinalWriteCode",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the distribution of final aggregated-answer semantic entropy "
            "estimated by repeatedly resampling random communication graphs per task."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        choices=sorted(DATASET_DEFAULTS.keys()),
        help="Dataset to run before plotting.",
    )
    parser.add_argument(
        "--dataset_json",
        type=str,
        default=None,
        help="Override the default JSON/JSONL dataset path for non-MMLU datasets.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["dev", "val", "test"],
        help="MMLU split. Ignored by non-MMLU datasets.",
    )
    parser.add_argument("--limit_questions", type=int, default=153)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of records inferred concurrently in each batch.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="Random",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star"],
        help="Communication graph topology. Use Random for the requested random graph setting.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Override dataset-specific default communication rounds.",
    )
    parser.add_argument(
        "--num_entropy_samples",
        type=int,
        default=5,
        help=(
            "Fresh graph+agent executions per task used to estimate final-answer "
            "semantic entropy."
        ),
    )
    # KLE temporarily disabled; keep this hyperparameter ready for future re-enable.
    # parser.add_argument(
    #     "--kle_heat_t",
    #     type=float,
    #     default=0.3,
    #     help="Heat-kernel lengthscale for KHEAT uncertainty.",
    # )
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Override the graph domain. Defaults to the selected dataset's domain.",
    )
    parser.add_argument("--decision_method", type=str, default=None)
    parser.add_argument("--agent_names", nargs="+", type=str, default=None)
    parser.add_argument("--agent_nums", nargs="+", type=int, default=None)
    parser.add_argument(
        "--semantic_judge_llm_name",
        type=str,
        default="",
        help="OpenAI-compatible semantic judge model name. Uses env defaults when omitted.",
    )
    parser.add_argument(
        "--semantic_judge_api_key",
        type=str,
        default="",
        help="API key for the semantic judge. Uses SEMANTIC_JUDGE_API_KEY or OPENAI_API_KEY when omitted.",
    )
    parser.add_argument(
        "--semantic_judge_base_url",
        type=str,
        default="",
        help="Base URL for the semantic judge API, e.g. a local vLLM OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--semantic_judge_model_path",
        type=str,
        default="",
        help="Optional local judge model path/name override.",
    )
    parser.add_argument(
        "--semantic_judge_max_concurrency",
        type=int,
        default=None,
        help="Maximum concurrent semantic judge requests.",
    )
    parser.add_argument("--humaneval_timeout", type=int, default=100)
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download/check MMLU data before loading MMLU.",
    )
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--csv_file", type=str, default=None)
    args = parser.parse_args()
    if args.agent_names is not None and args.agent_nums is not None:
        if len(args.agent_names) != len(args.agent_nums):
            parser.error("The number of agent names must match the number of agent counts.")
    elif args.agent_names is not None or args.agent_nums is not None:
        parser.error("--agent_names and --agent_nums must be provided together.")
    if args.num_rounds is not None and args.num_rounds < 1:
        parser.error("--num_rounds must be at least 1 when provided.")
    if args.num_entropy_samples < 1:
        parser.error("--num_entropy_samples must be at least 1.")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def first_answer(raw_answer: Any) -> str:
    if isinstance(raw_answer, list):
        raw_answer = raw_answer[0] if raw_answer else ""
    return raw_answer if isinstance(raw_answer, str) else str(raw_answer)


def numeric_correct(predicted: str, target: str) -> bool:
    try:
        return float(predicted) == float(target)
    except (TypeError, ValueError):
        return False


def choice_correct(predicted: str, target: str) -> bool:
    return predicted.strip().upper() == target.strip().upper()


def exact_correct(predicted: str, target: str) -> bool:
    return predicted == target


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def resolve_dataset_path(args, defaults: DatasetDefaults) -> str:
    path = args.dataset_json or defaults.dataset_json
    if path is None:
        raise ValueError(f"--dataset_json is required for dataset {args.dataset}.")
    return path


def resolve_dataset_bundle(args) -> DatasetBundle:
    defaults = DATASET_DEFAULTS[args.dataset]
    graph_domain = args.domain or defaults.graph_domain
    agent_names = args.agent_names or defaults.agent_names
    agent_nums = args.agent_nums or defaults.agent_nums
    num_rounds = args.num_rounds if args.num_rounds is not None else defaults.num_rounds
    decision_method = args.decision_method or defaults.decision_method

    if args.dataset == "mmlu":
        if args.download:
            download_mmlu()
        dataset = MMLUDataset(args.split)
        return DatasetBundle(
            name="mmlu",
            records=dataset,
            graph_domain=graph_domain,
            agent_names=agent_names,
            agent_nums=agent_nums,
            num_rounds=num_rounds,
            decision_method=decision_method,
            record_to_input=dataset.record_to_input,
            record_to_target=dataset.record_to_target_answer,
            parse_prediction=dataset.postprocess_answer,
            is_correct=exact_correct,
        )

    if args.dataset == "gsm8k":
        dataset = gsm_data_process(JSONLReader.parse_file(resolve_dataset_path(args, defaults)))
        return math_dataset_bundle(
            args.dataset,
            dataset,
            graph_domain,
            agent_names,
            agent_nums,
            num_rounds,
            decision_method,
            gsm_get_predict,
            numeric_correct,
        )

    if args.dataset == "aqua":
        dataset = aqua_data_process(JSONLReader.parse_file(resolve_dataset_path(args, defaults)))
        return math_dataset_bundle(
            args.dataset,
            dataset,
            graph_domain,
            agent_names,
            agent_nums,
            num_rounds,
            decision_method,
            aqua_get_predict,
            choice_correct,
        )

    if args.dataset == "svamp":
        dataset = svamp_data_process(JSONReader.parse_file(resolve_dataset_path(args, defaults)))
        return math_dataset_bundle(
            args.dataset,
            dataset,
            graph_domain,
            agent_names,
            agent_nums,
            num_rounds,
            decision_method,
            gsm_get_predict,
            numeric_correct,
        )

    if args.dataset == "multiarith":
        dataset = multiarith_data_process(JSONReader.parse_file(resolve_dataset_path(args, defaults)))
        return math_dataset_bundle(
            args.dataset,
            dataset,
            graph_domain,
            agent_names,
            agent_nums,
            num_rounds,
            decision_method,
            gsm_get_predict,
            numeric_correct,
        )

    if args.dataset == "humaneval":
        records = JSONLReader.parse_file(resolve_dataset_path(args, defaults))
        executor = PyExecutor()

        def humaneval_correct(predicted: str, target: str) -> bool:
            is_solved, _feedback, _state = executor.execute(
                predicted,
                [target],
                timeout=args.humaneval_timeout,
            )
            return bool(is_solved)

        return DatasetBundle(
            name="humaneval",
            records=records,
            graph_domain=graph_domain,
            agent_names=agent_names,
            agent_nums=agent_nums,
            num_rounds=num_rounds,
            decision_method=decision_method,
            record_to_input=lambda record: {"task": record["prompt"]},
            record_to_target=lambda record: record["test"],
            parse_prediction=lambda raw_answer: strip_code_fence(first_answer(raw_answer)),
            is_correct=humaneval_correct,
        )

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def math_dataset_bundle(
    name: str,
    records: Sequence[Dict[str, Any]],
    graph_domain: str,
    agent_names: List[str],
    agent_nums: List[int],
    num_rounds: int,
    decision_method: str,
    answer_parser: Callable[[str], str],
    correctness_fn: CorrectnessFn,
) -> DatasetBundle:
    return DatasetBundle(
        name=name,
        records=records,
        graph_domain=graph_domain,
        agent_names=agent_names,
        agent_nums=agent_nums,
        num_rounds=num_rounds,
        decision_method=decision_method,
        record_to_input=lambda record: {"task": record["task"]},
        record_to_target=lambda record: record["answer"],
        parse_prediction=lambda raw_answer: answer_parser(first_answer(raw_answer)),
        is_correct=correctness_fn,
    )


def get_graph_kwargs(mode: str, num_nodes: int) -> Dict[str, Any]:
    initial_spatial_probability = 0.5
    initial_temporal_probability = 0.5
    fixed_spatial_masks = None
    fixed_temporal_masks = None
    node_kwargs = None

    def generate_layered_graph(layer_num=2):
        adj_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        base_size = num_nodes // layer_num
        remainder = num_nodes % layer_num
        layers = []
        for layer_idx in range(layer_num):
            size = base_size + (1 if layer_idx < remainder else 0)
            layers.extend([layer_idx] * size)
        random.shuffle(layers)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if layers[j] == layers[i] + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph():
        matrix = [[0] * num_nodes for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                matrix[i][j] = 1
        return matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{"role": "Programming Expert"}]
    elif mode == "FullConnected":
        fixed_spatial_masks = [
            [1 if i != j else 0 for i in range(num_nodes)]
            for j in range(num_nodes)
        ]
        fixed_temporal_masks = [[1 for _ in range(num_nodes)] for _ in range(num_nodes)]
    elif mode == "Random":
        fixed_spatial_masks = [
            [random.randint(0, 1) if i != j else 0 for i in range(num_nodes)]
            for j in range(num_nodes)
        ]
        fixed_temporal_masks = [
            [random.randint(0, 1) for _ in range(num_nodes)]
            for _ in range(num_nodes)
        ]
    elif mode == "Chain":
        fixed_spatial_masks = [
            [1 if i == j + 1 else 0 for i in range(num_nodes)]
            for j in range(num_nodes)
        ]
        fixed_temporal_masks = [
            [1 if i == 0 and j == num_nodes - 1 else 0 for i in range(num_nodes)]
            for j in range(num_nodes)
        ]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        fixed_temporal_masks = [[1 for _ in range(num_nodes)] for _ in range(num_nodes)]
    elif mode == "Layered":
        fixed_spatial_masks = generate_layered_graph()
        fixed_temporal_masks = [[1 for _ in range(num_nodes)] for _ in range(num_nodes)]
    elif mode == "Star":
        fixed_spatial_masks = generate_star_graph()
        fixed_temporal_masks = [[1 for _ in range(num_nodes)] for _ in range(num_nodes)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }


def iter_batches(items: Iterable[Any], batch_size: int, limit: Optional[int]):
    batch = []
    for index, item in enumerate(items):
        if limit is not None and index >= limit:
            break
        batch.append((index, item))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def token_char_spans(token_logprobs: Sequence[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    reconstructed_parts = []
    spans = []
    cursor = 0
    for token_info in token_logprobs:
        token = str(attr_or_key(token_info, "token", ""))
        logprob = attr_or_key(token_info, "logprob")
        probability = attr_or_key(token_info, "probability")
        if logprob is not None:
            logprob = float(logprob)
        if probability is None and logprob is not None:
            probability = 0.0 if logprob < -745 else math.exp(logprob)

        start = cursor
        end = start + len(token)
        reconstructed_parts.append(token)
        spans.append(
            {
                "token": token,
                "start": start,
                "end": end,
                "logprob": logprob,
                "probability": probability,
            }
        )
        cursor = end
    return "".join(reconstructed_parts), spans


def substring_span(
    text: str,
    needle: str,
    start: int = 0,
    last: bool = False,
    case_sensitive: bool = True,
) -> Optional[AnswerSpan]:
    if not needle:
        return None
    haystack = text if case_sensitive else text.lower()
    target = needle if case_sensitive else needle.lower()
    index = haystack.rfind(target, start) if last else haystack.find(target, start)
    if index < 0:
        return None
    end = index + len(needle)
    return AnswerSpan(text=text[index:end], start=index, end=end)


def last_digit_span(text: str, answer: str, start: int = 0) -> Optional[AnswerSpan]:
    if not answer:
        return None
    matches = list(re.finditer(r"\d+", text[start:]))
    for match in reversed(matches):
        if match.group(0) == answer:
            span_start = start + match.start()
            span_end = start + match.end()
            return AnswerSpan(text=text[span_start:span_end], start=span_start, end=span_end)
    return None


def last_upper_letter_span(text: str, answer: str, start: int = 0) -> Optional[AnswerSpan]:
    if len(answer) != 1 or not answer.isupper():
        return None
    matches = list(re.finditer(r"[A-Z]", text[start:]))
    for match in reversed(matches):
        if match.group(0) == answer:
            span_start = start + match.start()
            span_end = start + match.end()
            return AnswerSpan(text=text[span_start:span_end], start=span_start, end=span_end)
    return None


def locate_mmlu_answer_span(text: str, answer: str) -> Optional[AnswerSpan]:
    marker = "answer is"
    marker_index = text.find(marker)
    if marker_index >= 0:
        marker_end = marker_index + len(marker)
        return substring_span(text, answer, marker_end)
    if text and answer and text[0] == answer:
        return AnswerSpan(text=text[0], start=0, end=1)
    return substring_span(text, answer)


def locate_math_answer_span(text: str, answer: str) -> Optional[AnswerSpan]:
    for marker in ("The answer is ", "the answer is "):
        marker_index = text.rfind(marker)
        if marker_index >= 0:
            marker_end = marker_index + len(marker)
            return (
                last_digit_span(text, answer, marker_end)
                or substring_span(text, answer, marker_end)
            )
    return last_digit_span(text, answer) or substring_span(text, answer, last=True)


def locate_aqua_answer_span(text: str, answer: str) -> Optional[AnswerSpan]:
    for marker in ("The answer is ", "the answer is "):
        marker_index = text.rfind(marker)
        if marker_index >= 0:
            marker_end = marker_index + len(marker)
            return (
                last_upper_letter_span(text, answer, marker_end)
                or substring_span(text, answer, marker_end)
            )
    return last_upper_letter_span(text, answer) or substring_span(text, answer, last=True)


def locate_humaneval_answer_span(text: str, answer: str) -> Optional[AnswerSpan]:
    exact_span = substring_span(text, answer)
    if exact_span is not None:
        return exact_span
    stripped_answer = answer.strip()
    if stripped_answer and stripped_answer != answer:
        return substring_span(text, stripped_answer)
    return None


def locate_extracted_answer_span(
    dataset_name: str,
    response_text: str,
    extracted_answer: str,
) -> Optional[AnswerSpan]:
    answer = "" if extracted_answer is None else str(extracted_answer)
    if not answer:
        return None
    if dataset_name == "mmlu":
        return locate_mmlu_answer_span(response_text, answer)
    if dataset_name in {"gsm8k", "svamp", "multiarith"}:
        return locate_math_answer_span(response_text, answer)
    if dataset_name == "aqua":
        return locate_aqua_answer_span(response_text, answer)
    if dataset_name == "humaneval":
        return locate_humaneval_answer_span(response_text, answer)
    return substring_span(response_text, answer)


def final_answer_logprob_stats(
    dataset_name: str,
    response_text: str,
    extracted_answer: str,
    token_logprobs: Sequence[Any],
) -> Dict[str, Any]:
    reconstructed_text, token_spans = token_char_spans(token_logprobs)
    span_text_source = reconstructed_text or response_text
    answer_span = locate_extracted_answer_span(
        dataset_name,
        span_text_source,
        extracted_answer,
    )
    stats = {
        "final_answer_span_text": answer_span.text if answer_span else "",
        "final_answer_span_start": answer_span.start if answer_span else "",
        "final_answer_span_end": answer_span.end if answer_span else "",
        "final_answer_token_count": 0,
        "final_answer_mean_logprob": None,
        "final_answer_mean_probability": None,
        "final_answer_logprob_uncertainty": None,
        "final_answer_logprob_tokens": [],
        "has_extracted_answer_span": answer_span is not None,
        "has_final_answer_logprobs": bool(token_spans),
    }
    if answer_span is None or not token_spans:
        return stats

    overlapping_tokens = [
        token_span
        for token_span in token_spans
        if token_span["end"] > answer_span.start and token_span["start"] < answer_span.end
        and token_span["logprob"] is not None
        and math.isfinite(token_span["logprob"])
    ]
    if not overlapping_tokens:
        return stats

    logprobs = [token_span["logprob"] for token_span in overlapping_tokens]
    mean_logprob = float(np.mean(logprobs))
    stats.update(
        {
            "final_answer_token_count": len(overlapping_tokens),
            "final_answer_mean_logprob": mean_logprob,
            "final_answer_mean_probability": float(math.exp(mean_logprob)),
            "final_answer_logprob_uncertainty": float(-mean_logprob),
            "final_answer_logprob_tokens": [
                {
                    "token": token_span["token"],
                    "logprob": token_span["logprob"],
                    "probability": token_span["probability"],
                }
                for token_span in overlapping_tokens
            ],
        }
    )
    return stats


async def compute_correctness(bundle: DatasetBundle, predicted: str, target: str) -> bool:
    if bundle.name == "humaneval":
        return await asyncio.to_thread(bundle.is_correct, predicted, target)
    return bundle.is_correct(predicted, target)


def build_graph(bundle: DatasetBundle, args, agent_names: List[str]) -> Graph:
    kwargs = get_graph_kwargs(args.mode, len(agent_names))
    graph = Graph(
        domain=bundle.graph_domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=bundle.decision_method,
        optimized_spatial=False,
        optimized_temporal=False,
        **kwargs,
    )
    graph.gcn.eval()
    graph.mlp.eval()
    return graph


async def run_graph_sample(
    bundle: DatasetBundle,
    input_dict: Dict[str, Any],
    args,
    agent_names: List[str],
) -> GraphSampleResult:
    realized_graph = build_graph(bundle, args, agent_names)
    raw_answer, _log_prob = await realized_graph.arun(
        input_dict,
        bundle.num_rounds,
        num_entropy_samples=1,
        record_execution_history=False,
        track_grad=False,
        record_decision_logprobs=False,
    )

    raw_final_answer = first_answer(raw_answer)
    predicted_answer = bundle.parse_prediction(raw_answer)
    return GraphSampleResult(
        raw_final_answer=raw_final_answer,
        predicted_answer=predicted_answer,
    )


def modal_answer(answers: Sequence[str]) -> str:
    valid_answers = [str(answer) for answer in answers if str(answer).strip()]
    if not valid_answers:
        return ""

    counts = Counter(valid_answers)
    best_answer = valid_answers[0]
    for answer in valid_answers[1:]:
        if counts[answer] > counts[best_answer]:
            best_answer = answer
    return best_answer


async def run_record_inference(
    bundle: DatasetBundle,
    record_index: int,
    record,
    args,
    agent_names: List[str],
) -> InferenceResult:
    input_dict = bundle.record_to_input(record)
    sample_results = await asyncio.gather(
        *[
            run_graph_sample(bundle, input_dict, args, agent_names)
            for _sample_idx in range(args.num_entropy_samples)
        ]
    )

    raw_final_answers = [sample.raw_final_answer for sample in sample_results]
    predicted_answers = [sample.predicted_answer for sample in sample_results]
    target_answer = bundle.record_to_target(record)
    modal_predicted_answer = modal_answer(predicted_answers)
    is_correct = await compute_correctness(bundle, modal_predicted_answer, target_answer)
    return InferenceResult(
        dataset=bundle.name,
        index=record_index,
        input_dict=input_dict,
        raw_final_answers=raw_final_answers,
        predicted_answers=predicted_answers,
        target_answer=target_answer,
        modal_predicted_answer=modal_predicted_answer,
        is_correct=is_correct,
    )


def exact_answer_semantic_entropy(outputs: Iterable[Any]) -> Tuple[float, Dict[str, Any]]:
    valid_outputs = [str(output).strip() for output in outputs if str(output).strip()]
    if len(valid_outputs) <= 1:
        return 0.0, {
            "method": "exact_answer_entropy",
            "outputs": valid_outputs,
            "labels": ["cluster_0"] if valid_outputs else [],
        }

    label_by_answer: Dict[str, str] = {}
    labels: List[str] = []
    for output in valid_outputs:
        normalized = " ".join(output.lower().split())
        if normalized not in label_by_answer:
            label_by_answer[normalized] = f"cluster_{len(label_by_answer)}"
        labels.append(label_by_answer[normalized])

    return semantic_entropy(labels), {
        "method": "exact_answer_entropy",
        "outputs": valid_outputs,
        "labels": labels,
    }


async def attach_final_answer_semantic_entropy(
    result: InferenceResult,
    judge: Optional[SemanticEntailmentJudge],
) -> Dict[str, Any]:
    question = str(result.input_dict.get("task", result.input_dict))
    if judge is not None and judge.is_configured:
        entropy, details = await semantic_uncertainty(
            question,
            result.predicted_answers,
            judge,
        )
    else:
        entropy, details = exact_answer_semantic_entropy(result.predicted_answers)

    return {
        "dataset": result.dataset,
        "index": result.index,
        "target_answer": result.target_answer,
        "modal_predicted_answer": result.modal_predicted_answer,
        "is_correct": result.is_correct,
        "num_graph_samples": len(result.predicted_answers),
        "semantic_entropy": entropy,
        "semantic_method": details.get("method", "semantic_entropy"),
        "semantic_labels": details.get("labels", []),
        "semantic_outputs": details.get("outputs", []),
        "predicted_answers": result.predicted_answers,
        "raw_final_answers": result.raw_final_answers,
    }


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "index",
                "target_answer",
                "modal_predicted_answer",
                "is_correct",
                "num_graph_samples",
                "semantic_entropy",
                "semantic_method",
                "semantic_labels",
                "semantic_outputs",
                "predicted_answers",
                "raw_final_answers",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "semantic_labels": repr(row["semantic_labels"]),
                    "semantic_outputs": repr(row["semantic_outputs"]),
                    "predicted_answers": repr(row["predicted_answers"]),
                    "raw_final_answers": repr(row["raw_final_answers"]),
                }
            )


def probability_weights(values: List[float]) -> np.ndarray:
    if not values:
        return np.array([])
    return np.ones(len(values), dtype=float) / len(values)


def plot_distribution(
    output_file: Path,
    dataset_name: str,
    mode_name: str,
    correct_values: List[float],
    incorrect_values: List[float],
    bins: int,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_values = correct_values + incorrect_values
    max_value = max(all_values) if all_values else 1.0
    if max_value <= 0:
        max_value = 1.0
    bin_edges = np.linspace(0.0, max_value, bins + 1)

    plt.figure(figsize=(7.0, 4.4))
    plt.hist(
        correct_values,
        bins=bin_edges,
        weights=probability_weights(correct_values),
        alpha=0.72,
        color="#4CAF50",
        label="Correct",
    )
    plt.hist(
        incorrect_values,
        bins=bin_edges,
        weights=probability_weights(incorrect_values),
        alpha=0.72,
        color="#FF5252",
        label="Incorrect",
    )
    plt.xlabel("Final Aggregated Answer Semantic Entropy (Random Graph Resampling)")
    plt.ylabel("Probability")
    plt.title(f"{dataset_name} {mode_name} Graph")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def default_output_paths(dataset_name: str, args) -> Tuple[Path, Path]:
    output_file = args.output_file or (
        f"result/{dataset_name}_{args.mode.lower()}_semantic_entropy_distribution.png"
    )
    csv_file = args.csv_file or (
        f"result/{dataset_name}_{args.mode.lower()}_semantic_entropy_distribution.csv"
    )
    return Path(output_file), Path(csv_file)


async def main():
    args = parse_args()
    seed_everything(args.seed)
    bundle = resolve_dataset_bundle(args)

    agent_names = [
        name for name, num in zip(bundle.agent_names, bundle.agent_nums) for _ in range(num)
    ]
    semantic_judge = SemanticEntailmentJudge(
        llm_name=args.semantic_judge_llm_name,
        api_key=args.semantic_judge_api_key,
        base_url=args.semantic_judge_base_url,
        model_path=args.semantic_judge_model_path,
        max_concurrency=args.semantic_judge_max_concurrency,
    )
    if not semantic_judge.is_configured:
        print("Semantic judge is not configured; falling back to exact extracted-answer entropy.")

    rows: List[Dict[str, Any]] = []
    correct_count = 0
    total_count = 0
    for batch_id, batch in enumerate(
        iter_batches(bundle.records, args.batch_size, args.limit_questions)
    ):
        print(f"Batch {batch_id}")
        inference_start = time.time()
        inference_results = await asyncio.gather(
            *[
                run_record_inference(bundle, record_index, record, args, agent_names)
                for record_index, record in batch
            ]
        )
        inference_seconds = time.time() - inference_start

        entropy_start = time.time()
        batch_rows = await asyncio.gather(
            *[
                attach_final_answer_semantic_entropy(inference_result, semantic_judge)
                for inference_result in inference_results
            ]
        )
        entropy_seconds = time.time() - entropy_start

        rows.extend(batch_rows)
        correct_count += sum(int(row["is_correct"]) for row in batch_rows)
        total_count += len(batch_rows)
        print(f"Accuracy so far: {correct_count / total_count:.4f}")
        print(f"Inference time: {inference_seconds:.3f}s")
        print(f"Semantic entropy time: {entropy_seconds:.3f}s")

    correct_values = [
        row["semantic_entropy"]
        for row in rows
        if row["is_correct"]
        and row["semantic_entropy"] is not None
        and math.isfinite(row["semantic_entropy"])
    ]
    incorrect_values = [
        row["semantic_entropy"]
        for row in rows
        if not row["is_correct"]
        and row["semantic_entropy"] is not None
        and math.isfinite(row["semantic_entropy"])
    ]

    output_file, csv_file = default_output_paths(bundle.name, args)
    save_csv(csv_file, rows)
    plot_distribution(
        output_file,
        bundle.name,
        args.mode,
        correct_values,
        incorrect_values,
        args.bins,
    )
    print(f"Saved CSV: {csv_file}")
    print(f"Saved plot: {output_file}")
    print(f"Correct samples: {len(correct_values)}")
    print(f"Incorrect samples: {len(incorrect_values)}")
    print(f"Rows with semantic entropy: {len(correct_values) + len(incorrect_values)} / {len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())
