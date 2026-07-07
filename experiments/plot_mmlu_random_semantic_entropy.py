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
from datasets.MMLU.download import download as download_mmlu
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from datasets.gsm8k_dataset import (
    gsm_data_process,
    gsm_get_predict,
    multiarith_data_process,
    svamp_data_process,
)
from datasets.mmlu_dataset import MMLUDataset


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
    mean_agent_topk_token_entropy: Optional[float]
    agent_topk_token_entropy_details: List[Dict[str, Any]]


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
            "Plot the distribution of mean agent uncertainty from top-k token "
            "entropy over the first generated tokens in one sampled graph per task."
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
        "--topk",
        type=int,
        default=5,
        help="Number of top token candidates used to compute each token entropy.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="Number of leading generated tokens used for each agent output.",
    )
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
    if args.topk < 1:
        parser.error("--topk must be at least 1.")
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
    graph.gcn.eval()
    graph.mlp.eval()
    return graph


def top_logprob_entropy(token_info: Any, topk: int) -> Optional[float]:
    top_entries = attr_or_key(token_info, "top_logprobs", []) or []
    logprobs: List[float] = []
    for entry in top_entries[:topk]:
        logprob = attr_or_key(entry, "logprob")
        if logprob is None:
            continue
        logprob = float(logprob)
        if math.isfinite(logprob):
            logprobs.append(logprob)

    if not logprobs:
        return None

    max_logprob = max(logprobs)
    exp_values = [math.exp(logprob - max_logprob) for logprob in logprobs]
    normalizer = sum(exp_values)
    if normalizer <= 0:
        return None

    probabilities = [value / normalizer for value in exp_values]
    return float(
        -sum(probability * math.log(probability) for probability in probabilities if probability > 0)
    )


def output_topk_token_entropy(
    token_logprobs: Sequence[Any],
    k_tokens: int,
    topk: int,
) -> Dict[str, Any]:
    token_entropies = []
    for token_info in token_logprobs[:k_tokens]:
        entropy = top_logprob_entropy(token_info, topk)
        if entropy is not None and math.isfinite(entropy):
            token_entropies.append(entropy)

    if not token_entropies:
        return {
            "entropy": None,
            "tokens_used": 0,
            "token_entropies": [],
        }

    return {
        "entropy": float(np.mean(token_entropies)),
        "tokens_used": len(token_entropies),
        "token_entropies": token_entropies,
    }


def collect_agent_topk_token_entropy(
    graph: Graph,
    k_tokens: int,
    topk: int,
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
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
                stats = output_topk_token_entropy(token_logprobs, k_tokens, topk)
                details.append(
                    {
                        "node_id": node.id,
                        "agent_name": node.agent_name,
                        "role": node.role,
                        "round": history.get("round", ""),
                        "output_index": output_index,
                        "entropy": stats["entropy"],
                        "tokens_used": stats["tokens_used"],
                        "token_entropies": stats["token_entropies"],
                    }
                )

    valid_entropies = [
        detail["entropy"]
        for detail in details
        if detail["entropy"] is not None and math.isfinite(detail["entropy"])
    ]
    if not valid_entropies:
        return None, details
    return float(np.mean(valid_entropies)), details


async def run_record_inference(
    bundle: DatasetBundle,
    record_index: int,
    record,
    args,
    agent_names: List[str],
) -> InferenceResult:
    input_dict = bundle.record_to_input(record)
    realized_graph = build_graph(bundle, args, agent_names)
    raw_answer, _log_prob = await realized_graph.arun(
        input_dict,
        bundle.num_rounds,
        num_entropy_samples=1,
        record_execution_history=True,
        track_grad=False,
        record_node_logprobs=True,
        node_top_logprobs=args.topk,
        record_decision_logprobs=False,
    )

    raw_final_answer = first_answer(raw_answer)
    predicted_answer = bundle.parse_prediction(raw_answer)
    target_answer = bundle.record_to_target(record)
    is_correct = await compute_correctness(bundle, predicted_answer, target_answer)
    mean_entropy, entropy_details = collect_agent_topk_token_entropy(
        realized_graph,
        args.k,
        args.topk,
    )
    return InferenceResult(
        dataset=bundle.name,
        index=record_index,
        input_dict=input_dict,
        raw_final_answer=raw_final_answer,
        predicted_answer=predicted_answer,
        target_answer=target_answer,
        is_correct=is_correct,
        mean_agent_topk_token_entropy=mean_entropy,
        agent_topk_token_entropy_details=entropy_details,
    )


async def attach_agent_topk_token_uncertainty(
    result: InferenceResult,
) -> Dict[str, Any]:
    return {
        "dataset": result.dataset,
        "index": result.index,
        "raw_final_answer": result.raw_final_answer,
        "predicted_answer": result.predicted_answer,
        "target_answer": result.target_answer,
        "is_correct": result.is_correct,
        "mean_agent_topk_token_entropy": result.mean_agent_topk_token_entropy,
        "agent_topk_token_entropy_details": result.agent_topk_token_entropy_details,
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
                "mean_agent_topk_token_entropy",
                "agent_topk_token_entropy_details",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "agent_topk_token_entropy_details": repr(
                        row["agent_topk_token_entropy_details"]
                    ),
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
    plt.xlabel("Mean Agent Top-k Token Entropy (First K Tokens)")
    plt.ylabel("Probability")
    plt.title(f"{dataset_name} {mode_name} Graph")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def default_output_paths(dataset_name: str, args) -> Tuple[Path, Path]:
    output_file = args.output_file or (
        f"result/{dataset_name}_{args.mode.lower()}_agent_topk_token_entropy_distribution.png"
    )
    csv_file = args.csv_file or (
        f"result/{dataset_name}_{args.mode.lower()}_agent_topk_token_entropy_distribution.csv"
    )
    return Path(output_file), Path(csv_file)


async def main():
    args = parse_args()
    seed_everything(args.seed)
    bundle = resolve_dataset_bundle(args)

    agent_names = [
        name for name, num in zip(bundle.agent_names, bundle.agent_nums) for _ in range(num)
    ]

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

        uncertainty_start = time.time()
        batch_rows = await asyncio.gather(
            *[
                attach_agent_topk_token_uncertainty(inference_result)
                for inference_result in inference_results
            ]
        )
        uncertainty_seconds = time.time() - uncertainty_start

        rows.extend(batch_rows)
        correct_count += sum(int(row["is_correct"]) for row in batch_rows)
        total_count += len(batch_rows)
        print(f"Accuracy so far: {correct_count / total_count:.4f}")
        print(f"Inference time: {inference_seconds:.3f}s")
        print(f"Top-k token entropy time: {uncertainty_seconds:.3f}s")

    correct_values = [
        row["mean_agent_topk_token_entropy"]
        for row in rows
        if row["is_correct"]
        and row["mean_agent_topk_token_entropy"] is not None
        and math.isfinite(row["mean_agent_topk_token_entropy"])
    ]
    incorrect_values = [
        row["mean_agent_topk_token_entropy"]
        for row in rows
        if not row["is_correct"]
        and row["mean_agent_topk_token_entropy"] is not None
        and math.isfinite(row["mean_agent_topk_token_entropy"])
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
    print(f"Rows with top-k token entropy: {len(correct_values) + len(incorrect_values)} / {len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())
