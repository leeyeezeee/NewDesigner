import argparse
import asyncio
import csv
import math
import os
import random
import re
import sys
import time
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
from GDesigner.utils.ig_scorer import FinalAnswerScorer, make_target_spec
from GDesigner.utils.uncertainty import edge_entropy_rewards
from datasets.MMLU.download import download as download_mmlu
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from datasets.gsm8k_dataset import (
    gsm_data_process,
    gsm_get_predict,
    multiarith_data_process,
    svamp_data_process,
)
from datasets.mmlu_dataset import MMLUDataset
from experiments.agent_backend import add_agent_backend_args, apply_agent_backend_args
from experiments.graph_concurrency import limited_graph_arun, make_graph_semaphore


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
class InferenceResult:
    dataset: str
    index: int
    input_dict: Dict[str, Any]
    raw_final_answer: str
    predicted_answer: str
    target_answer: str
    is_correct: bool
    graph_edge_ig_sum: float
    graph_edge_ig_count: int
    edge_ig_details: List[Dict[str, Any]]


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
            "Plot the relationship between the graph edge information-gain sum "
            "and final-answer correctness in one sampled graph per task."
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
        "--k",
        type=int,
        default=16,
        help="Deprecated; edge IG no longer uses generated token prefixes.",
    )
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    add_agent_backend_args(parser)
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Override the graph domain. Defaults to the selected dataset's domain.",
    )
    parser.add_argument("--decision_method", type=str, default=None)
    parser.add_argument("--agent_names", nargs="+", type=str, default=None)
    parser.add_argument("--agent_nums", nargs="+", type=int, default=None)
    parser.add_argument("--humaneval_timeout", type=int, default=100)
    parser.add_argument(
        "--max_concurrent_graphs",
        type=int,
        default=10,
        help=(
            "Maximum number of realized graphs to execute concurrently per batch. "
            "Use 0 or a negative value for unlimited concurrency."
        ),
    )
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
    if args.k < 1:
        parser.error("--k must be at least 1.")
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
    graph.gat.eval()
    graph.spatial_affinity.eval()
    return graph


def token_probability_from_logprob(logprob: Optional[float]) -> Optional[float]:
    if logprob is None:
        return None
    logprob = float(logprob)
    if not math.isfinite(logprob):
        return None
    return 0.0 if logprob < -745 else float(math.exp(logprob))


def output_token_logprob_probability_stats(
    token_logprobs: Sequence[Any],
    k_tokens: int,
) -> Dict[str, Any]:
    logprobs = []
    probabilities = []
    tokens = []
    for token_info in token_logprobs[:k_tokens]:
        logprob = attr_or_key(token_info, "logprob")
        probability = attr_or_key(token_info, "probability")
        if probability is None:
            probability = token_probability_from_logprob(logprob)
        if logprob is None or probability is None:
            continue
        logprob = float(logprob)
        probability = float(probability)
        if not math.isfinite(logprob) or not math.isfinite(probability):
            continue
        tokens.append(str(attr_or_key(token_info, "token", "")))
        logprobs.append(logprob)
        probabilities.append(probability)

    if not logprobs:
        return {
            "mean_logprob": None,
            "mean_probability": None,
            "tokens_used": 0,
            "tokens": [],
            "token_logprobs": [],
            "token_probabilities": [],
        }

    return {
        "mean_logprob": float(np.mean(logprobs)),
        "mean_probability": float(np.mean(probabilities)),
        "tokens_used": len(logprobs),
        "tokens": tokens,
        "token_logprobs": logprobs,
        "token_probabilities": probabilities,
    }


def collect_agent_token_logprob_probability(
    graph: Graph,
    k_tokens: int,
) -> Tuple[Optional[float], Optional[float], List[Dict[str, Any]]]:
    details = []
    for node in graph.nodes.values():
        histories = node.execution_history or [
            {
                "round": "",
                "output_token_logprobs": getattr(node, "output_token_logprobs", []),
            }
        ]
        for history in histories:
            for output_index, token_logprobs in enumerate(
                history.get("output_token_logprobs", [])
            ):
                stats = output_token_logprob_probability_stats(token_logprobs, k_tokens)
                details.append(
                    {
                        "node_id": node.id,
                        "agent_name": node.agent_name,
                        "role": node.role,
                        "round": history.get("round", ""),
                        "output_index": output_index,
                        "mean_logprob": stats["mean_logprob"],
                        "mean_probability": stats["mean_probability"],
                        "tokens_used": stats["tokens_used"],
                        "tokens": stats["tokens"],
                        "token_logprobs": stats["token_logprobs"],
                        "token_probabilities": stats["token_probabilities"],
                    }
                )

    valid_logprobs = [
        detail["mean_logprob"]
        for detail in details
        if detail["mean_logprob"] is not None and math.isfinite(detail["mean_logprob"])
    ]
    valid_probabilities = [
        detail["mean_probability"]
        for detail in details
        if detail["mean_probability"] is not None
        and math.isfinite(detail["mean_probability"])
    ]
    mean_logprob = float(np.mean(valid_logprobs)) if valid_logprobs else None
    mean_probability = float(np.mean(valid_probabilities)) if valid_probabilities else None
    return mean_logprob, mean_probability, details


def target_spec_for_bundle(bundle: DatasetBundle, target_answer: str):
    if bundle.name == "humaneval":
        return make_target_spec(bundle.name, tests=[target_answer])
    return make_target_spec(bundle.name, target_answer)


def observed_edge_infos_from_history(graph: Graph) -> List[Dict[str, Any]]:
    edge_infos: List[Dict[str, Any]] = []
    seen = set()
    for target_id, node in graph.nodes.items():
        for history in node.execution_history:
            round_idx = history.get("round")
            if round_idx is None:
                continue
            for edge_type, info_key in (
                ("spatial", "spatial_info"),
                ("temporal", "temporal_info"),
            ):
                for source_id in history.get(info_key, {}).keys():
                    key = f"{edge_type}:{round_idx}:{source_id}->{target_id}"
                    if key in seen:
                        continue
                    seen.add(key)
                    edge_infos.append(
                        {
                            "type": edge_type,
                            "round": round_idx,
                            "source": source_id,
                            "target": target_id,
                            "edge_key": key,
                        }
                    )
    return edge_infos


def edge_details_to_rows(edge_details: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for key, detail in sorted(edge_details.items()):
        ig_gain = detail.get("ig_gain")
        rows.append(
            {
                "edge_key": key,
                "type": detail.get("type"),
                "round": detail.get("round"),
                "source": detail.get("source"),
                "target": detail.get("target"),
                "ig_gain": float(ig_gain) if ig_gain is not None else None,
                "ig_mode": detail.get("ig_mode"),
                "uncertainty_method": detail.get("uncertainty_method"),
                "teacher_forcing_agent": detail.get("teacher_forcing_agent"),
                "teacher_forcing_candidate_role": detail.get("teacher_forcing_candidate_role"),
                "scoring_agent": detail.get("scoring_agent"),
                "execution_candidate_role": detail.get("execution_candidate_role"),
                "before_teacher_logprob": detail.get("before_teacher_logprob"),
                "after_teacher_logprob": detail.get("after_teacher_logprob"),
                "before_answer_score": detail.get("before_answer_score"),
                "after_answer_score": detail.get("after_answer_score"),
            }
        )
    return rows


async def compute_graph_edge_ig_sum(
    graph: Graph,
    bundle: DatasetBundle,
    input_dict: Dict[str, Any],
    target_answer: str,
    scorer: FinalAnswerScorer,
) -> Tuple[float, int, List[Dict[str, Any]]]:
    graph.edge_log_probs = observed_edge_infos_from_history(graph)
    if not graph.edge_log_probs:
        return 0.0, 0, []

    # Edge IG replaces the receiver output, then scores the final agent context.
    _edge_rewards, edge_details = await edge_entropy_rewards(
        graph,
        input_dict.get("task", str(input_dict)),
        input_dict,
        judge=None,
        num_entropy_samples=1,
        target_spec=target_spec_for_bundle(bundle, target_answer),
        ig_scorer=scorer,
        compute_rewards=False,
    )
    detail_rows = edge_details_to_rows(edge_details)
    gains = [
        row["ig_gain"]
        for row in detail_rows
        if row["ig_gain"] is not None and math.isfinite(row["ig_gain"])
    ]
    return float(sum(gains)), len(gains), detail_rows


async def run_record_inference(
    bundle: DatasetBundle,
    record_index: int,
    record,
    args,
    agent_names: List[str],
    graph_semaphore: asyncio.Semaphore | None,
) -> InferenceResult:
    input_dict = bundle.record_to_input(record)
    realized_graph = build_graph(bundle, args, agent_names)
    scorer = FinalAnswerScorer()
    raw_answer, _log_prob = await limited_graph_arun(
        graph_semaphore,
        realized_graph,
        input_dict,
        bundle.num_rounds,
        num_entropy_samples=1,
        record_execution_history=True,
        track_grad=False,
        record_node_logprobs=False,
        node_logprob_token_limit=None,
        record_decision_logprobs=False,
    )

    raw_final_answer = first_answer(raw_answer)
    predicted_answer = bundle.parse_prediction(raw_answer)
    target_answer = bundle.record_to_target(record)
    is_correct = await compute_correctness(bundle, predicted_answer, target_answer)
    edge_ig_sum, edge_ig_count, edge_ig_details = await compute_graph_edge_ig_sum(
        realized_graph,
        bundle,
        input_dict,
        target_answer,
        scorer,
    )
    return InferenceResult(
        dataset=bundle.name,
        index=record_index,
        input_dict=input_dict,
        raw_final_answer=raw_final_answer,
        predicted_answer=predicted_answer,
        target_answer=target_answer,
        is_correct=is_correct,
        graph_edge_ig_sum=edge_ig_sum,
        graph_edge_ig_count=edge_ig_count,
        edge_ig_details=edge_ig_details,
    )


async def attach_edge_ig_summary(
    result: InferenceResult,
) -> Dict[str, Any]:
    return {
        "dataset": result.dataset,
        "index": result.index,
        "raw_final_answer": result.raw_final_answer,
        "predicted_answer": result.predicted_answer,
        "target_answer": result.target_answer,
        "is_correct": result.is_correct,
        "graph_edge_ig_sum": result.graph_edge_ig_sum,
        "graph_edge_ig_count": result.graph_edge_ig_count,
        "edge_ig_details": result.edge_ig_details,
    }


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "index",
                "raw_final_answer",
                "predicted_answer",
                "target_answer",
                "is_correct",
                "graph_edge_ig_sum",
                "graph_edge_ig_count",
                "edge_ig_details",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "edge_ig_details": repr(row["edge_ig_details"]),
                }
            )


def plot_edge_ig_correctness_relationship(
    output_file: Path,
    dataset_name: str,
    mode_name: str,
    rows: List[Dict[str, Any]],
    bins: int,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    valid_rows = [
        row
        for row in rows
        if row["graph_edge_ig_sum"] is not None
        and math.isfinite(float(row["graph_edge_ig_sum"]))
    ]
    correct_values = [
        float(row["graph_edge_ig_sum"])
        for row in valid_rows
        if row["is_correct"]
    ]
    incorrect_values = [
        float(row["graph_edge_ig_sum"])
        for row in valid_rows
        if not row["is_correct"]
    ]

    plt.figure(figsize=(7.0, 4.4))
    if correct_values:
        plt.scatter(
            correct_values,
            [1.0] * len(correct_values),
            alpha=0.72,
            color="#4CAF50",
            label="Correct",
        )
    if incorrect_values:
        plt.scatter(
            incorrect_values,
            [0.0] * len(incorrect_values),
            alpha=0.72,
            color="#FF5252",
            label="Incorrect",
        )

    all_values = correct_values + incorrect_values
    if len(set(all_values)) > 1:
        bin_count = max(1, min(int(bins), len(set(all_values))))
        bin_edges = np.linspace(min(all_values), max(all_values), bin_count + 1)
        bin_centers = []
        bin_accuracies = []
        for left, right in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = [
                row
                for row in valid_rows
                if float(row["graph_edge_ig_sum"]) >= left
                and (
                    float(row["graph_edge_ig_sum"]) < right
                    or right == bin_edges[-1]
                    and float(row["graph_edge_ig_sum"]) <= right
                )
            ]
            if not in_bin:
                continue
            bin_centers.append((left + right) / 2.0)
            bin_accuracies.append(
                sum(1.0 for row in in_bin if row["is_correct"]) / len(in_bin)
            )
        if bin_centers:
            plt.plot(
                bin_centers,
                bin_accuracies,
                color="#1F77B4",
                linewidth=1.8,
                marker="o",
                markersize=3.2,
                label="Binned Accuracy",
            )

    plt.xlabel("图的边信息增益加和")
    plt.ylabel("Correctness")
    plt.yticks([0, 1], ["Incorrect", "Correct"])
    plt.title(f"{dataset_name} {mode_name} Graph")
    plt.legend()
    plt.grid(axis="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def default_output_paths(dataset_name: str, args) -> Tuple[Path, Path]:
    output_file = args.output_file or (
        f"result/{dataset_name}_{args.mode.lower()}_edge_ig_sum_correctness.png"
    )
    csv_file = args.csv_file or (
        f"result/{dataset_name}_{args.mode.lower()}_edge_ig_sum_correctness.csv"
    )
    return Path(output_file), Path(csv_file)


async def main():
    args = parse_args()
    apply_agent_backend_args(args)
    seed_everything(args.seed)
    bundle = resolve_dataset_bundle(args)

    agent_names = [
        name for name, num in zip(bundle.agent_names, bundle.agent_nums) for _ in range(num)
    ]

    rows: List[Dict[str, Any]] = []
    correct_count = 0
    total_count = 0
    graph_semaphore = make_graph_semaphore(args.max_concurrent_graphs)
    for batch_id, batch in enumerate(
        iter_batches(bundle.records, args.batch_size, args.limit_questions)
    ):
        print(f"Batch {batch_id}")
        inference_start = time.time()
        inference_results = await asyncio.gather(
            *[
                run_record_inference(bundle, record_index, record, args, agent_names, graph_semaphore)
                for record_index, record in batch
            ]
        )
        inference_seconds = time.time() - inference_start

        uncertainty_start = time.time()
        batch_rows = await asyncio.gather(
            *[
                attach_edge_ig_summary(inference_result)
                for inference_result in inference_results
            ]
        )
        uncertainty_seconds = time.time() - uncertainty_start

        rows.extend(batch_rows)
        correct_count += sum(int(row["is_correct"]) for row in batch_rows)
        total_count += len(batch_rows)
        print(f"Accuracy so far: {correct_count / total_count:.4f}")
        print(f"Inference time: {inference_seconds:.3f}s")
        print(f"Edge IG time: {uncertainty_seconds:.3f}s")

    output_file, csv_file = default_output_paths(bundle.name, args)
    save_csv(csv_file, rows)
    plot_edge_ig_correctness_relationship(
        output_file,
        bundle.name,
        args.mode,
        rows,
        args.bins,
    )
    rows_with_edge_ig = [
        row
        for row in rows
        if row["graph_edge_ig_count"] > 0
        and row["graph_edge_ig_sum"] is not None
        and math.isfinite(row["graph_edge_ig_sum"])
    ]
    correct_values = [row["graph_edge_ig_sum"] for row in rows_with_edge_ig if row["is_correct"]]
    incorrect_values = [row["graph_edge_ig_sum"] for row in rows_with_edge_ig if not row["is_correct"]]
    print(f"Saved CSV: {csv_file}")
    print(f"Saved plot: {output_file}")
    print(f"Correct samples: {len(correct_values)}")
    print(f"Incorrect samples: {len(incorrect_values)}")
    print(f"Rows with edge IG: {len(rows_with_edge_ig)} / {len(rows)}")


if __name__ == "__main__":
    from experiments.crash_logging import run_async_with_crash_logging
    run_async_with_crash_logging(main)
