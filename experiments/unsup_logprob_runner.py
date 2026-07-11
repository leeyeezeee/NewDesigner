import asyncio
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import torch

from GDesigner.graph.graph import Graph
from GDesigner.utils.globals import CompletionTokens, Cost, PromptTokens, Time
from GDesigner.utils.metrics import reset_usage_counters, write_metrics_record
from experiments.checkpoint import load_graph_checkpoint, save_graph_checkpoint
from experiments.graph_concurrency import (
    limited_async_call,
    limited_graph_arun,
    make_graph_semaphore,
)
from experiments.plot_mmlu_random_semantic_entropy import (
    final_answer_logprob_stats,
    first_answer,
    strip_code_fence,
)


AnswerParser = Callable[[str], str]
CorrectnessFn = Callable[[str, str], bool]
RecordToInput = Callable[[Any], Dict[str, Any]]
RecordToTarget = Callable[[Any], str]


@dataclass
class UnsupSampleResult:
    reward: Optional[float]
    uncertainty: Optional[float]
    graph_log_prob: torch.Tensor
    reference_kl: torch.Tensor
    raw_answer: str
    predicted_answer: str


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Unsupervised data not found: {data_path}")
    records = []
    with data_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_json(path: str) -> List[Dict[str, Any]]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with data_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else [data]


def limit_records(records: Sequence[Any], limit: Optional[int]) -> List[Any]:
    if limit is None or limit <= 0:
        return list(records)
    return list(records[:limit])


def dataloader(records: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def numeric_correct(predicted: str, target: str) -> bool:
    try:
        return float(predicted) == float(target)
    except (TypeError, ValueError):
        return False


def choice_correct(predicted: str, target: str) -> bool:
    return predicted.strip().upper() == target.strip().upper()


def exact_correct(predicted: str, target: str) -> bool:
    return predicted == target


def unlabeled_record_to_input(dataset_name: str, record: Dict[str, Any]) -> Dict[str, str]:
    if "task" in record:
        return {"task": str(record["task"])}
    if dataset_name == "mmlu":
        task = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
        )
        return {"task": task}
    if dataset_name == "aqua":
        options = " ".join(str(option) for option in record.get("options", []))
        return {"task": f"{record['question']} Choices: {options}"}
    if dataset_name == "svamp":
        return {"task": f"{record['Body'].strip()} {record['Question'].strip()}"}
    if dataset_name == "multiarith":
        return {"task": str(record["sQuestion"]).strip()}
    if dataset_name == "humaneval":
        return {"task": str(record["prompt"])}
    if "question" in record:
        return {"task": str(record["question"])}
    raise ValueError(f"Cannot build task input for {dataset_name}: {record}")


def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
        Literal["Mesh"],
    ],
    num_nodes: int,
):
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

    def generate_mesh_graph():
        return [[1 if i != j else 0 for i in range(num_nodes)] for j in range(num_nodes)]

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
    elif mode == "Mesh":
        fixed_spatial_masks = generate_mesh_graph()
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


def build_graph(args, graph_domain: str) -> Graph:
    agent_names = [
        name
        for name, num in zip(args.agent_names, args.agent_nums)
        for _ in range(num)
    ]
    kwargs = get_kwargs(args.mode, len(agent_names))
    return Graph(
        domain=graph_domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        refine_rank=args.refine_rank,
        edge_bias_scale=args.edge_bias_scale,
        **kwargs,
    )


def share_trainable_graph_state(realized_graph: Graph, graph: Graph) -> None:
    realized_graph.gcn = graph.gcn
    realized_graph.mlp = graph.mlp
    realized_graph.spatial_affinity_weight = graph.spatial_affinity_weight
    realized_graph.refinement_weight = graph.refinement_weight
    realized_graph.spatial_edge_bias = graph.spatial_edge_bias
    realized_graph.temporal_logits = graph.temporal_logits


def configure_train_scope(graph: Graph, train_scope: str) -> List[torch.nn.Parameter]:
    train_gnn = train_scope == "all"
    train_refinement = train_scope in {"bias_refinement", "all"}
    train_bias = train_scope in {"bias", "bias_refinement", "all"}

    for parameter in graph.gcn.parameters():
        parameter.requires_grad = train_gnn
    for parameter in graph.mlp.parameters():
        parameter.requires_grad = train_gnn
    graph.spatial_affinity_weight.requires_grad = train_refinement
    graph.refinement_weight.requires_grad = train_refinement
    graph.spatial_edge_bias.requires_grad = train_bias
    graph.temporal_logits.requires_grad = graph.optimized_temporal and train_scope == "all"

    parameters: List[torch.nn.Parameter] = []
    if train_gnn:
        parameters.extend(graph.gcn.parameters())
        parameters.extend(graph.mlp.parameters())
    if train_refinement:
        parameters.append(graph.spatial_affinity_weight)
        parameters.append(graph.refinement_weight)
    if train_bias:
        parameters.append(graph.spatial_edge_bias)
    if graph.temporal_logits.requires_grad:
        parameters.append(graph.temporal_logits)
    if not parameters:
        raise ValueError(f"No trainable parameters selected by --train_scope {train_scope!r}.")
    return parameters


def spatial_probabilities(graph: Graph) -> torch.Tensor:
    probabilities = graph.spatial_edge_probabilities
    if probabilities is None:
        probabilities = torch.sigmoid(graph.spatial_logits)
    mask = graph.spatial_masks.to(device=probabilities.device, dtype=torch.bool)
    return probabilities.view(-1)[mask.view(-1)].clamp(1e-6, 1.0 - 1e-6)


def bernoulli_reference_kl(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    current = current.clamp(1e-6, 1.0 - 1e-6)
    reference = reference.to(device=current.device, dtype=current.dtype).clamp(1e-6, 1.0 - 1e-6)
    return (
        current * (current.log() - reference.log())
        + (1.0 - current) * ((1.0 - current).log() - (1.0 - reference).log())
    ).mean()


def reference_spatial_kl(
    realized_graph: Graph,
    reference_graph: Graph,
    input_dict: Dict[str, str],
) -> torch.Tensor:
    current_probabilities = spatial_probabilities(realized_graph)
    with torch.no_grad():
        reference_graph.prepare_spatial_logits(input_dict["task"], track_grad=False)
        reference_probabilities = spatial_probabilities(reference_graph).detach()
    return bernoulli_reference_kl(current_probabilities, reference_probabilities)


def fallback_mean_logprob(token_logprobs: Sequence[Any]) -> Optional[float]:
    values = [
        float(token.logprob)
        for token in token_logprobs
        if getattr(token, "logprob", None) is not None
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def final_answer_reward(
    dataset_name: str,
    raw_answer: Any,
    token_logprobs: Sequence[Any],
    answer_parser: AnswerParser,
) -> Tuple[Optional[float], Optional[float], str, str]:
    raw_text = first_answer(raw_answer)
    predicted_answer = answer_parser(raw_text)
    stats = final_answer_logprob_stats(
        dataset_name,
        raw_text,
        predicted_answer,
        list(token_logprobs),
    )
    mean_logprob = stats.get("final_answer_mean_logprob")
    if mean_logprob is None:
        mean_logprob = fallback_mean_logprob(token_logprobs)
    if mean_logprob is None:
        return None, None, raw_text, predicted_answer
    reward = float(mean_logprob)
    return reward, float(-reward), raw_text, predicted_answer


async def run_unsup_sample(
    graph: Graph,
    reference_graph: Graph,
    input_dict: Dict[str, str],
    args,
    dataset_name: str,
    answer_parser: AnswerParser,
) -> UnsupSampleResult:
    realized_graph = copy.deepcopy(graph)
    share_trainable_graph_state(realized_graph, graph)
    raw_answer, graph_log_prob = await realized_graph.arun(
        input_dict,
        args.num_rounds,
        num_entropy_samples=1,
        record_execution_history=False,
        track_grad=True,
        record_decision_logprobs=True,
    )
    token_logprobs = list(
        getattr(realized_graph.decision_node, "last_response_token_logprobs", [])
    )
    reward, uncertainty, raw_text, predicted_answer = final_answer_reward(
        dataset_name,
        raw_answer,
        token_logprobs,
        answer_parser,
    )
    return UnsupSampleResult(
        reward=reward,
        uncertainty=uncertainty,
        graph_log_prob=graph_log_prob,
        reference_kl=reference_spatial_kl(realized_graph, reference_graph, input_dict),
        raw_answer=raw_text,
        predicted_answer=predicted_answer,
    )


async def train_unsup_logprob(
    graph: Graph,
    reference_graph: Graph,
    args,
    dataset_name: str,
    unlabeled_records: Sequence[Dict[str, Any]],
    answer_parser: AnswerParser,
) -> Dict[str, Any]:
    graph.gcn.train()
    graph.mlp.train()
    reference_graph.gcn.eval()
    reference_graph.mlp.eval()

    parameters = configure_train_scope(graph, args.train_scope)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    graph_semaphore = make_graph_semaphore(args.max_concurrent_graphs)

    total_groups = 0
    total_valid_samples = 0
    last_loss = 0.0
    last_policy_loss = 0.0
    last_reference_kl = 0.0
    last_mean_uncertainty = None

    for epoch in range(args.unsup_epochs):
        print(f"Unsup epoch {epoch}", 80 * "-")
        records = list(unlabeled_records)
        random.shuffle(records)
        max_batches = math.ceil(len(records) / args.batch_size)
        for batch_id, batch in enumerate(dataloader(records, args.batch_size)):
            if args.num_iterations is not None and batch_id >= args.num_iterations:
                break
            start_ts = time.time()
            group_tasks = []
            for record in batch:
                input_dict = unlabeled_record_to_input(dataset_name, record)
                group_tasks.append([
                    asyncio.create_task(
                        limited_async_call(
                            graph_semaphore,
                            run_unsup_sample,
                            graph,
                            reference_graph,
                            input_dict,
                            args,
                            dataset_name,
                            answer_parser,
                        )
                    )
                    for _ in range(args.graph_sample_count)
                ])

            group_results = [await asyncio.gather(*tasks) for tasks in group_tasks]
            policy_losses = []
            reference_kls = []
            uncertainties = []
            valid_groups = 0
            valid_samples = 0
            for results in group_results:
                valid_results = [result for result in results if result.reward is not None]
                if len(valid_results) < 2:
                    continue
                rewards = torch.tensor(
                    [result.reward for result in valid_results],
                    dtype=torch.float32,
                )
                baseline = float(rewards.mean().item())
                valid_groups += 1
                valid_samples += len(valid_results)
                for result in valid_results:
                    advantage = result.reward - baseline
                    policy_losses.append(-result.graph_log_prob * advantage)
                    reference_kls.append(result.reference_kl)
                    if result.uncertainty is not None:
                        uncertainties.append(result.uncertainty)

            if not policy_losses:
                print(
                    f"Batch {batch_id}/{max_batches}: no valid answer logprobs; "
                    "skipping optimizer step."
                )
                continue

            policy_loss = torch.mean(torch.stack(policy_losses))
            reference_kl = torch.mean(torch.stack(reference_kls))
            total_loss = policy_loss + args.reference_kl_weight * reference_kl

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_groups += valid_groups
            total_valid_samples += valid_samples
            last_loss = float(total_loss.item())
            last_policy_loss = float(policy_loss.item())
            last_reference_kl = float(reference_kl.item())
            last_mean_uncertainty = (
                float(sum(uncertainties) / len(uncertainties))
                if uncertainties
                else None
            )
            print(
                f"Batch {batch_id}/{max_batches}: "
                f"loss={last_loss:.6f}, "
                f"policy={last_policy_loss:.6f}, "
                f"ref_kl={last_reference_kl:.6f}, "
                f"mean_uncertainty={last_mean_uncertainty}, "
                f"valid_groups={valid_groups}, "
                f"valid_samples={valid_samples}, "
                f"time={time.time() - start_ts:.3f}s"
            )

    return {
        "unsup_groups": total_groups,
        "unsup_valid_samples": total_valid_samples,
        "unsup_loss": last_loss,
        "unsup_policy_loss": last_policy_loss,
        "unsup_reference_kl": last_reference_kl,
        "unsup_mean_uncertainty": last_mean_uncertainty,
    }


async def evaluate_records(
    graph: Graph,
    records: Sequence[Any],
    args,
    *,
    record_to_input: RecordToInput,
    record_to_target: RecordToTarget,
    answer_parser: AnswerParser,
    correctness_fn: CorrectnessFn,
    dataset_name: str,
) -> Dict[str, Any]:
    graph.gcn.eval()
    graph.mlp.eval()
    total_solved = 0.0
    total_executed = 0
    total_edges = 0
    edge_samples = 0
    graph_semaphore = make_graph_semaphore(args.max_concurrent_graphs)
    eval_records = list(records)
    if args.eval_skip:
        eval_records = eval_records[args.eval_skip:]
    if args.eval_limit is not None and args.eval_limit > 0:
        eval_records = eval_records[:args.eval_limit]

    for batch_id, batch in enumerate(dataloader(eval_records, args.batch_size)):
        print(f"Eval batch {batch_id}", 80 * "-")
        start_ts = time.time()
        tasks = []
        realized_graphs = []
        for record in batch:
            realized_graph = copy.deepcopy(graph)
            share_trainable_graph_state(realized_graph, graph)
            realized_graphs.append(realized_graph)
            tasks.append(asyncio.create_task(
                limited_graph_arun(
                    graph_semaphore,
                    realized_graph,
                    record_to_input(record),
                    args.num_rounds,
                    num_entropy_samples=1,
                    record_execution_history=False,
                    track_grad=False,
                )
            ))
        raw_results = await asyncio.gather(*tasks)
        raw_answers, _log_probs = zip(*raw_results)
        for realized_graph in realized_graphs:
            total_edges += sum(realized_graph.realized_edge_counts)
            edge_samples += 1
        for raw_answer, record in zip(raw_answers, batch):
            predicted = answer_parser(first_answer(raw_answer))
            target = record_to_target(record)
            if dataset_name == "humaneval":
                is_solved = await asyncio.to_thread(correctness_fn, predicted, target)
            else:
                is_solved = correctness_fn(predicted, target)
            total_solved += float(is_solved)
            total_executed += 1
        accuracy = total_solved / total_executed if total_executed else 0.0
        print(f"Eval batch time {time.time() - start_ts:.3f}s")
        print(f"Accuracy: {accuracy}")
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    return {
        "accuracy": total_solved / total_executed if total_executed else 0.0,
        "total_solved": total_solved,
        "total_executed": total_executed,
        "avg_edges": total_edges / edge_samples if edge_samples else 0.0,
    }


async def run_unsup_stage(
    args,
    *,
    dataset_name: str,
    graph_domain: str,
    eval_records: Sequence[Any],
    record_to_input: RecordToInput,
    record_to_target: RecordToTarget,
    answer_parser: AnswerParser,
    correctness_fn: CorrectnessFn,
) -> None:
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    unlabeled_records = limit_records(load_jsonl(args.unsup_data), args.unsup_limit)
    if not unlabeled_records:
        raise ValueError(f"No unsupervised records loaded from {args.unsup_data}")

    graph = build_graph(args, graph_domain)
    reference_graph = build_graph(args, graph_domain)
    load_graph_checkpoint(graph, args.stage1_checkpoint)
    load_graph_checkpoint(reference_graph, args.stage1_checkpoint)
    for parameter in reference_graph.gcn.parameters():
        parameter.requires_grad = False
    for parameter in reference_graph.mlp.parameters():
        parameter.requires_grad = False
    reference_graph.refinement_weight.requires_grad = False
    reference_graph.spatial_affinity_weight.requires_grad = False
    reference_graph.spatial_edge_bias.requires_grad = False
    reference_graph.temporal_logits.requires_grad = False

    train_metrics = await train_unsup_logprob(
        graph,
        reference_graph,
        args,
        dataset_name,
        unlabeled_records,
        answer_parser,
    )

    reset_usage_counters()
    print("Start Eval")
    eval_metrics = await evaluate_records(
        graph,
        eval_records,
        args,
        record_to_input=record_to_input,
        record_to_target=record_to_target,
        answer_parser=answer_parser,
        correctness_fn=correctness_fn,
        dataset_name=dataset_name,
    )
    metrics = {
        **train_metrics,
        **eval_metrics,
    }
    save_graph_checkpoint(
        graph,
        args.checkpoint_file,
        dataset=dataset_name,
        args=args,
        metrics=metrics,
    )
    print(f"Final Eval Accuracy: {eval_metrics['accuracy']}")
    print(f"Final Avg Edges: {eval_metrics['avg_edges']}")
    write_metrics_record(args.metrics_file, {
        "dataset": dataset_name,
        "accuracy": eval_metrics["accuracy"],
        "total_solved": eval_metrics["total_solved"],
        "total_executed": eval_metrics["total_executed"],
        "avg_edges": eval_metrics["avg_edges"],
        "llm_name": args.llm_name,
        "stage": "unsup_logprob",
        "unsup_groups": train_metrics["unsup_groups"],
        "unsup_valid_samples": train_metrics["unsup_valid_samples"],
        "unsup_reference_kl": train_metrics["unsup_reference_kl"],
    })


def add_common_unsup_args(parser, *, dataset_name: str, unsup_data: str, stage1_checkpoint: str, checkpoint_file: str, metrics_file: str) -> None:
    parser.add_argument("--unsup_data", type=str, default=unsup_data)
    parser.add_argument("--unsup_limit", type=int, default=128)
    parser.add_argument("--unsup_epochs", type=int, default=1)
    parser.add_argument("--graph_sample_count", type=int, default=4)
    parser.add_argument(
        "--max_concurrent_graphs",
        type=int,
        default=10,
        help=(
            "Maximum number of realized graphs to execute concurrently per batch. "
            "Use 0 or a negative value for unlimited concurrency."
        ),
    )
    parser.add_argument("--reference_kl_weight", type=float, default=0.05)
    parser.add_argument(
        "--train_scope",
        type=str,
        default="bias_refinement",
        choices=["bias", "bias_refinement", "all"],
    )
    parser.add_argument("--stage1_checkpoint", type=str, default=stage1_checkpoint)
    parser.add_argument("--checkpoint_file", type=str, default=checkpoint_file)
    parser.add_argument("--metrics_file", type=str, default=metrics_file)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument(
        "--stage1_num_iterations",
        type=int,
        default=10,
        help="Used to preserve the original train/eval split for non-MMLU datasets.",
    )
    parser.add_argument(
        "--stage1_batch_size",
        type=int,
        default=4,
        help="Used with --stage1_num_iterations to preserve the original eval split.",
    )
    parser.add_argument("--eval_skip", type=int, default=None)
    parser.set_defaults(optimized_spatial=True, optimized_temporal=False)
    parser.add_argument("--no_optimized_spatial", action="store_false", dest="optimized_spatial")
    parser.add_argument("--optimized_temporal", action="store_true")
    parser.set_defaults(_unsup_dataset_name=dataset_name)


def finalize_unsup_args(parser, args) -> Any:
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    if args.graph_sample_count < 2:
        parser.error("--graph_sample_count must be at least 2 to build an advantage.")
    if args.unsup_limit is not None and args.unsup_limit < 1:
        parser.error("--unsup_limit must be at least 1.")
    if args.unsup_epochs < 1:
        parser.error("--unsup_epochs must be at least 1.")
    if args.reference_kl_weight < 0:
        parser.error("--reference_kl_weight must be non-negative.")
    if not args.optimized_spatial:
        parser.error("The unsupervised logprob stage requires optimized spatial edges.")
    if args.eval_skip is None:
        if args._unsup_dataset_name == "mmlu":
            args.eval_skip = 0
        else:
            args.eval_skip = int(args.stage1_num_iterations) * int(args.stage1_batch_size)
    return args


def humaneval_answer_parser(text: str) -> str:
    return strip_code_fence(text)
