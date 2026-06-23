import asyncio
import copy
import os
import random
import time
from typing import Callable, Dict, List, Literal, Tuple, Union

import torch

from GDesigner.graph.graph import Graph
from GDesigner.utils.globals import CompletionTokens, Cost, PromptTokens, Time
from GDesigner.utils.metrics import reset_usage_counters, write_metrics_record
from GDesigner.utils.edge_selector import (
    EdgeSelector,
    SelectorReplayBuffer,
    build_edge_selector_examples,
    train_edge_selector,
)
from GDesigner.utils.uncertainty import (
    SemanticEntailmentJudge,
    edge_entropy_rewards,
)


AnswerParser = Callable[[str], str]
CorrectnessFn = Callable[[str, str], bool]


def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch * batch_size : i_batch * batch_size + batch_size]


def numeric_correct(predicted: str, target: str) -> bool:
    try:
        return float(predicted) == float(target)
    except (TypeError, ValueError):
        return False


def choice_correct(predicted: str, target: str) -> bool:
    return predicted.strip().upper() == target.strip().upper()


async def run_math_dataset(
    args,
    dataset,
    *,
    dataset_name: str,
    graph_domain: str,
    answer_parser: AnswerParser,
    correctness_fn: CorrectnessFn,
) -> None:
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    agent_names = [
        name
        for name, num in zip(args.agent_names, args.agent_nums)
        for _ in range(num)
    ]
    kwargs = get_kwargs(args.mode, len(agent_names))
    graph = Graph(
        domain=graph_domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        **kwargs,
    )
    graph.gcn.train()
    graph.mlp.train()
    optimizer_params = list(graph.gcn.parameters()) + list(graph.mlp.parameters())
    if graph.optimized_temporal:
        optimizer_params.append(graph.temporal_logits)
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)

    effective_num_entropy_samples = (
        max(2, int(args.num_entropy_samples))
        if args.use_edge_selector
        else max(1, int(args.num_entropy_samples))
    )
    optimize_enabled = args.optimized_spatial or args.optimized_temporal
    use_semantic_edges_for_analysis = (
        optimize_enabled
        and args.use_edge_selector
        and effective_num_entropy_samples > 1
    )
    semantic_judge = None
    if use_semantic_edges_for_analysis:
        semantic_judge = SemanticEntailmentJudge(
            llm_name=args.semantic_judge_llm_name,
            api_key=args.semantic_judge_api_key,
            base_url=args.semantic_judge_base_url,
            model_path=args.semantic_judge_model_path,
            max_concurrency=args.semantic_judge_max_concurrency,
        )
    edge_selector = None
    selector_buffer = None
    selector_optimizer = None
    selector_trained = False
    if use_semantic_edges_for_analysis:
        edge_selector = EdgeSelector(graph.features.size(1))
        selector_buffer = SelectorReplayBuffer(getattr(args, "selector_buffer_size", 512))
        selector_optimizer = torch.optim.Adam(edge_selector.parameters(), lr=1e-3)

    num_batches = int(len(dataset) / args.batch_size)
    total_solved, total_executed = (0, 0)
    total_edges, edge_samples = (0, 0)
    accuracy = 0.0

    for i_batch in range(num_batches):
        train_updates_enabled = optimize_enabled and i_batch < args.num_iterations
        use_semantic_edges = use_semantic_edges_for_analysis and train_updates_enabled
        batch_entropy_samples = effective_num_entropy_samples if use_semantic_edges else 1
        batch_edge_selector = edge_selector if (selector_trained and not train_updates_enabled) else None
        print(f"Batch {i_batch}", 80 * "-")
        start_ts = time.time()
        answer_log_probs = []
        answers = []
        realized_graphs = []
        input_dicts = []

        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if current_batch is None:
            print("No more data available.")
            break

        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            realized_graph.temporal_logits = graph.temporal_logits
            realized_graphs.append(realized_graph)
            task = record["task"]
            answers.append(record["answer"])
            input_dict = {"task": task}
            input_dicts.append(input_dict)
            answer_log_probs.append(
                asyncio.create_task(
                    realized_graph.arun(
                        input_dict,
                        args.num_rounds,
                        num_entropy_samples=batch_entropy_samples,
                        record_execution_history=use_semantic_edges,
                        track_grad=train_updates_enabled,
                        edge_selector=batch_edge_selector,
                    )
                )
            )

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[dict] = []

        for record, answer, log_prob, true_answer, realized_graph, input_dict in zip(
            current_batch,
            raw_answers,
            log_probs,
            answers,
            realized_graphs,
            input_dicts,
        ):
            predict_answer = answer_parser(answer[0])
            is_solved = correctness_fn(predict_answer, true_answer)
            correctness_reward = float(is_solved)
            total_solved += int(is_solved)
            total_executed += 1
            accuracy = total_solved / total_executed
            if not train_updates_enabled:
                total_edges += sum(realized_graph.realized_edge_counts)
                edge_samples += 1

            edge_rewards = {}
            edge_details = {}
            if (
                is_solved
                and use_semantic_edges
            ):
                edge_rewards, edge_details = await edge_entropy_rewards(
                    realized_graph,
                    record["task"],
                    input_dict,
                    semantic_judge,
                    effective_num_entropy_samples,
                    negative_reward_scale=args.negative_edge_reward_scale,
                    nonpositive_penalty=args.nonpositive_edge_penalty,
                )
                selector_buffer.add_many(build_edge_selector_examples(
                    realized_graph,
                    record["task"],
                    edge_details,
                    getattr(args, "selector_entropy_tau", 0.0),
                ))
            realized_graph.clear_execution_history()
            utility = {
                "correctness": correctness_reward,
                "edge_entropy_rewards": edge_rewards,
            }
            utilities.append(utility)
            single_loss = -log_prob * correctness_reward
            loss_list.append(single_loss)

        total_loss = torch.mean(torch.stack(loss_list))
        if train_updates_enabled:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if edge_selector is not None:
                selector_trained = (
                    train_edge_selector(edge_selector, selector_optimizer, selector_buffer)
                    or selector_trained
                )
            if (
                graph.optimized_temporal
                and (i_batch + 1) % args.imp_per_iterations == 0
            ):
                temporal_masks, pruned_temporal_idx = graph.prune_temporal_edges(args.pruning_rate)
                print(f"pruned temporal edges: {pruned_temporal_idx.numel()}")
                print("temporal masks:", temporal_masks.view(graph.num_nodes, graph.num_nodes))

        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print("utilities:", utilities)
        print("loss:", total_loss.item())

        if i_batch + 1 == args.num_iterations:
            total_solved = 0
            total_executed = 0
            total_edges = 0
            edge_samples = 0
            accuracy = 0.0
            graph.gcn.eval()
            graph.mlp.eval()
            reset_usage_counters()
            print("Start Eval")

        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    print(f"Final Eval Accuracy: {accuracy}")
    print(f"Final Cost {Cost.instance().value}")
    print(f"Final PromptTokens {PromptTokens.instance().value}")
    print(f"Final CompletionTokens {CompletionTokens.instance().value}")
    avg_edges = total_edges / edge_samples if edge_samples else 0.0
    print(f"Final Avg Edges {avg_edges}")
    write_metrics_record(args.metrics_file, {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "total_solved": total_solved,
        "total_executed": total_executed,
        "avg_edges": avg_edges,
        "llm_name": args.llm_name,
    })


def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
    ],
    N: int,
):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    def generate_layered_graph(num_nodes, layer_num=2):
        adj_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        base_size = num_nodes // layer_num
        remainder = num_nodes % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(num_nodes):
            current_layer = layers[i]
            for j in range(num_nodes):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(num_nodes):
        matrix = [[0] * num_nodes for _ in range(num_nodes)]
        for i in range(0, num_nodes):
            for j in range(i + 1, num_nodes):
                matrix[i][j] = 1
        return matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{"role": "Programming Expert"}]
    elif mode == "FullConnected":
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Random":
        fixed_spatial_masks = [
            [random.randint(0, 1) if i != j else 0 for i in range(N)]
            for j in range(N)
        ]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == "Chain":
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Layered":
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Star":
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }
