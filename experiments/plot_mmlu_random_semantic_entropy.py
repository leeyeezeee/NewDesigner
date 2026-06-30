import argparse
import asyncio
import copy
import csv
import os
import random
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
from GDesigner.utils.uncertainty import SemanticEntailmentJudge, semantic_uncertainty
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
    decision_method: str


@dataclass
class DatasetBundle:
    name: str
    records: Sequence[Any]
    graph_domain: str
    agent_names: List[str]
    agent_nums: List[int]
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
    realized_graph: Graph
    predicted_answer: str
    target_answer: str
    is_correct: bool


DATASET_DEFAULTS: Dict[str, DatasetDefaults] = {
    "mmlu": DatasetDefaults(
        dataset_json=None,
        graph_domain="mmlu",
        agent_names=["AnalyzeAgent"],
        agent_nums=[5],
        decision_method="FinalRefer",
    ),
    "gsm8k": DatasetDefaults(
        dataset_json="datasets/gsm8k/gsm8k.jsonl",
        graph_domain="gsm8k",
        agent_names=["MathSolver"],
        agent_nums=[4],
        decision_method="FinalRefer",
    ),
    "aqua": DatasetDefaults(
        dataset_json="datasets/AQuA/AQuA.jsonl",
        graph_domain="aqua",
        agent_names=["MathSolver_aqua"],
        agent_nums=[4],
        decision_method="FinalRefer",
    ),
    "svamp": DatasetDefaults(
        dataset_json="datasets/SVAMP/SVAMP.json",
        graph_domain="gsm8k",
        agent_names=["MathSolver"],
        agent_nums=[4],
        decision_method="FinalRefer",
    ),
    "multiarith": DatasetDefaults(
        dataset_json="datasets/MultiArith/MultiArith.json",
        graph_domain="gsm8k",
        agent_names=["MathSolver"],
        agent_nums=[4],
        decision_method="FinalRefer",
    ),
    "humaneval": DatasetDefaults(
        dataset_json="datasets/humaneval/humaneval-py.jsonl",
        graph_domain="humaneval",
        agent_names=["CodeWriting"],
        agent_nums=[5],
        decision_method="FinalWriteCode",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the post-communication distribution of average agent semantic "
            "entropy for correct vs incorrect samples."
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
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument(
        "--num_entropy_samples",
        type=int,
        default=4,
        help="Samples per agent used to estimate KHEAT uncertainty. Use at least 2.",
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
    parser.add_argument("--semantic_judge_llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--semantic_judge_api_key", type=str, default="")
    parser.add_argument("--semantic_judge_base_url", type=str, default="")
    parser.add_argument("--semantic_judge_model_path", type=str, default="")
    parser.add_argument(
        "--semantic_judge_max_concurrency",
        type=int,
        default=None,
        help="Maximum concurrent semantic judge requests. Defaults to env or 16.",
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
    if args.num_entropy_samples < 2:
        parser.error("--num_entropy_samples must be at least 2 to estimate KHEAT uncertainty.")
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


def decision_agent_nodes(graph) -> List[Any]:
    participants = [
        node for node in graph.decision_node.spatial_predecessors if node.id in graph.nodes
    ]
    return participants or list(graph.nodes.values())


async def average_agent_semantic_entropy(
    graph,
    question: str,
    judge: SemanticEntailmentJudge,
    heat_t: float = 0.3,
) -> Tuple[float, Dict[str, float]]:
    agents = decision_agent_nodes(graph)
    entropy_tasks = []
    entropy_nodes = []

    for node in agents:
        samples = list(node.entropy_samples) or list(node.outputs)
        samples = [str(sample) for sample in samples if str(sample).strip()]
        if not samples:
            continue
        entropy_nodes.append(node)
        entropy_tasks.append(semantic_uncertainty(question, samples, judge, heat_t=heat_t))

    if not entropy_tasks:
        return 0.0, {}

    entropy_results = await asyncio.gather(*entropy_tasks)
    per_agent_entropy = {
        node.id: float(entropy)
        for node, (entropy, _labels) in zip(entropy_nodes, entropy_results)
    }
    average_entropy = float(np.mean(list(per_agent_entropy.values())))
    return average_entropy, per_agent_entropy


async def compute_correctness(bundle: DatasetBundle, predicted: str, target: str) -> bool:
    if bundle.name == "humaneval":
        return await asyncio.to_thread(bundle.is_correct, predicted, target)
    return bundle.is_correct(predicted, target)


async def run_record_inference(
    graph: Graph,
    bundle: DatasetBundle,
    record_index: int,
    record,
    args,
) -> InferenceResult:
    realized_graph = copy.deepcopy(graph)
    realized_graph.gcn = graph.gcn
    realized_graph.mlp = graph.mlp
    realized_graph.temporal_logits = graph.temporal_logits

    input_dict = bundle.record_to_input(record)
    raw_answer, _log_prob = await realized_graph.arun(
        input_dict,
        args.num_rounds,
        num_entropy_samples=args.num_entropy_samples,
        record_execution_history=False,
        track_grad=False,
    )

    predicted_answer = bundle.parse_prediction(raw_answer)
    target_answer = bundle.record_to_target(record)
    is_correct = await compute_correctness(bundle, predicted_answer, target_answer)
    return InferenceResult(
        dataset=bundle.name,
        index=record_index,
        input_dict=input_dict,
        realized_graph=realized_graph,
        predicted_answer=predicted_answer,
        target_answer=target_answer,
        is_correct=is_correct,
    )


async def attach_semantic_entropy(
    result: InferenceResult,
    judge: SemanticEntailmentJudge,
    heat_t: float = 0.3,
) -> Dict[str, Any]:
    avg_entropy, per_agent_entropy = await average_agent_semantic_entropy(
        result.realized_graph,
        result.input_dict["task"],
        judge,
        heat_t=heat_t,
    )

    return {
        "dataset": result.dataset,
        "index": result.index,
        "predicted_answer": result.predicted_answer,
        "target_answer": result.target_answer,
        "is_correct": result.is_correct,
        "average_agent_semantic_entropy": avg_entropy,
        "per_agent_semantic_entropy": per_agent_entropy,
    }


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "index",
                "predicted_answer",
                "target_answer",
                "is_correct",
                "average_agent_semantic_entropy",
                "per_agent_semantic_entropy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "per_agent_semantic_entropy": repr(row["per_agent_semantic_entropy"]),
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
    plt.xlabel("Average Agent KHEAT Uncertainty")
    plt.ylabel("Probability")
    plt.title(f"{dataset_name} {mode_name} Graph")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.gca().invert_xaxis()
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

    judge = SemanticEntailmentJudge(
        llm_name=args.semantic_judge_llm_name,
        api_key=args.semantic_judge_api_key,
        base_url=args.semantic_judge_base_url,
        model_path=args.semantic_judge_model_path,
        max_concurrency=args.semantic_judge_max_concurrency,
    )
    if not judge.is_configured:
        raise RuntimeError(
            "Semantic judge is not configured. Set OPENAI_API_KEY or pass "
            "--semantic_judge_api_key; for local vLLM, pass "
            "--semantic_judge_base_url http://localhost:8000/v1."
        )
    print(f"Semantic judge max concurrency: {judge.max_concurrency}")

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
                run_record_inference(graph, bundle, record_index, record, args)
                for record_index, record in batch
            ]
        )
        inference_seconds = time.time() - inference_start

        entropy_start = time.time()
        batch_rows = await asyncio.gather(
            *[
                attach_semantic_entropy(inference_result, judge)
                for inference_result in inference_results
            ]
        )
        entropy_seconds = time.time() - entropy_start

        rows.extend(batch_rows)
        correct_count += sum(int(row["is_correct"]) for row in batch_rows)
        total_count += len(batch_rows)
        print(f"Accuracy so far: {correct_count / total_count:.4f}")
        print(f"Inference time: {inference_seconds:.3f}s")
        print(f"KHEAT uncertainty time: {entropy_seconds:.3f}s")

    correct_values = [
        row["average_agent_semantic_entropy"] for row in rows if row["is_correct"]
    ]
    incorrect_values = [
        row["average_agent_semantic_entropy"] for row in rows if not row["is_correct"]
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


if __name__ == "__main__":
    asyncio.run(main())
