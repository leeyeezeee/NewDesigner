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
from experiments.agent_backend import apply_agent_backend_args
from GDesigner.utils.edge_selector import (
    EdgeSelector,
    SelectorReplayBuffer,
    build_edge_selector_examples,
    train_edge_selector,
)
from GDesigner.utils.uncertainty import (
    edge_entropy_rewards,
)
from GDesigner.utils.ig_scorer import FinalAnswerScorer, make_target_spec
from experiments.checkpoint import save_graph_checkpoint
from experiments.graph_concurrency import limited_graph_arun, make_graph_semaphore
from experiments.teacher_forcing_reward import (
    edge_information_gain_loss,
    graph_correctness_advantage_edge_loss,
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
    apply_agent_backend_args(args)
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
        refine_rank=getattr(args, "refine_rank", 4),
        **kwargs,
    )
    graph.gat.train()
    optimizer_params = list(graph.gat.parameters()) + graph.refinement_parameters()
    if graph.optimized_temporal:
        optimizer_params.append(graph.temporal_logits)
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)

    use_graph_tf_reward = bool(getattr(args, "use_graph_tf_reward", False))
    use_graph_correctness_advantage = bool(
        getattr(args, "use_graph_correctness_advantage", False)
    )
    raw_edge_ig_reward_lambda = getattr(args, "edge_ig_reward_lambda", None)
    edge_ig_reward_lambda = (
        0.0
        if raw_edge_ig_reward_lambda is None
        else float(raw_edge_ig_reward_lambda)
    )
    use_multi_graph_reward = use_graph_tf_reward or use_graph_correctness_advantage
    effective_num_entropy_samples = 1
    optimize_enabled = args.optimized_spatial or args.optimized_temporal
    use_semantic_edges_for_analysis = (
        optimize_enabled
        and (
            args.use_edge_selector
            or edge_ig_reward_lambda != 0.0
        )
    )
    semantic_judge = None
    edge_selector = None
    selector_buffer = None
    selector_optimizer = None
    selector_trained = False
    if args.use_edge_selector and use_semantic_edges_for_analysis:
        edge_selector = EdgeSelector(graph.features.size(1))
        selector_buffer = SelectorReplayBuffer(getattr(args, "selector_buffer_size", 512))
        selector_optimizer = torch.optim.Adam(edge_selector.parameters(), lr=1e-3)
    tf_scorer = (
        FinalAnswerScorer()
        if (args.use_edge_selector or edge_ig_reward_lambda != 0.0)
        else None
    )
    graph_semaphore = make_graph_semaphore(getattr(args, "max_concurrent_graphs", 10))

    num_batches = int(len(dataset) / args.batch_size)
    total_solved, total_executed = (0, 0)
    total_edges, edge_samples = (0, 0)
    accuracy = 0.0

    for i_batch in range(num_batches):
        train_updates_enabled = optimize_enabled and i_batch < args.num_iterations
        use_semantic_edges = use_semantic_edges_for_analysis and train_updates_enabled
        batch_entropy_samples = 1
        batch_edge_selector = edge_selector if (selector_trained and not train_updates_enabled) else None
        print(f"Batch {i_batch}", 80 * "-")
        start_ts = time.time()
        answer_log_probs = []
        answers = []
        realized_graphs = []
        input_dicts = []
        record_input_dicts = []
        sample_groups = []

        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if current_batch is None:
            print("No more data available.")
            break

        for record in current_batch:
            task = record["task"]
            answers.append(record["answer"])
            input_dict = {"task": task}
            record_input_dicts.append(input_dict)
            group_indices = []
            sample_count = (
                max(1, int(getattr(args, "graph_sample_count", 5)))
                if (use_multi_graph_reward and train_updates_enabled)
                else 1
            )
            for _ in range(sample_count):
                realized_graph = copy.deepcopy(graph)
                realized_graph.gat = graph.gat
                realized_graph.spatial_affinity_weight = graph.spatial_affinity_weight
                realized_graph.refinement_weight = graph.refinement_weight
                realized_graph.temporal_logits = graph.temporal_logits
                group_indices.append(len(realized_graphs))
                realized_graphs.append(realized_graph)
                input_dicts.append(input_dict)
                answer_log_probs.append(
                    asyncio.create_task(
                        limited_graph_arun(
                            graph_semaphore,
                            realized_graph,
                            input_dict,
                            args.num_rounds,
                            num_entropy_samples=batch_entropy_samples,
                            record_execution_history=use_semantic_edges,
                            track_grad=train_updates_enabled,
                            edge_selector=batch_edge_selector,
                        )
                    )
                )
            sample_groups.append(group_indices)

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[dict] = []

        if use_multi_graph_reward and train_updates_enabled:
            graph_groups = []
            graph_log_prob_groups = []
            correctness_groups = []
            edge_detail_groups = []
            graph_tf_corrects: List[float] = []
            graph_tf_edge_counts: List[float] = []
            for record, true_answer, group_indices, input_dict in zip(
                current_batch,
                answers,
                sample_groups,
                record_input_dicts,
            ):
                target_spec = make_target_spec(dataset_name, true_answer)
                graph_group = []
                graph_log_prob_group = []
                correctness_group = []
                edge_detail_group = []
                for sample_pos, graph_idx in enumerate(group_indices):
                    realized_graph = realized_graphs[graph_idx]
                    answer = raw_answers[graph_idx]
                    predict_answer = answer_parser(answer[0])
                    is_solved = correctness_fn(predict_answer, true_answer)
                    correctness_reward = float(is_solved)
                    graph_tf_corrects.append(correctness_reward)
                    graph_tf_edge_counts.append(float(sum(realized_graph.realized_edge_counts)))
                    needs_edge_details = (
                        bool(realized_graph.edge_log_probs)
                        and (
                            edge_ig_reward_lambda != 0.0
                            or selector_buffer is not None
                        )
                    )
                    if needs_edge_details:
                        _edge_rewards, edge_details = await edge_entropy_rewards(
                            realized_graph,
                            record["task"],
                            input_dict,
                            semantic_judge,
                            effective_num_entropy_samples,
                            negative_reward_scale=args.negative_edge_reward_scale,
                            nonpositive_penalty=args.nonpositive_edge_penalty,
                            kle_heat_t=getattr(args, "kle_heat_t", 0.3),
                            target_spec=target_spec,
                            ig_scorer=tf_scorer,
                            compute_rewards=False,
                        )
                    else:
                        edge_details = {}
                    if selector_buffer is not None:
                        selector_buffer.add_many(build_edge_selector_examples(
                            realized_graph,
                            record["task"],
                            edge_details,
                            getattr(args, "selector_ig_tau", 0.0),
                        ))
                    if sample_pos == 0:
                        total_solved += int(is_solved)
                        total_executed += 1
                        accuracy = total_solved / total_executed
                        utilities.append({
                            "correctness": correctness_reward,
                        })
                    graph_group.append(realized_graph)
                    graph_log_prob_group.append(log_probs[graph_idx])
                    correctness_group.append(correctness_reward)
                    edge_detail_group.append(edge_details)
                    realized_graph.clear_execution_history()
                graph_groups.append(graph_group)
                graph_log_prob_groups.append(graph_log_prob_group)
                correctness_groups.append(correctness_group)
                edge_detail_groups.append(edge_detail_group)
            reference_loss = torch.mean(torch.stack(list(log_probs)))
            utility_loss, tf_summaries = graph_correctness_advantage_edge_loss(
                graph_groups,
                graph_log_prob_groups,
                correctness_groups,
                edge_detail_groups,
                reference_loss,
                edge_tanh_temperature=getattr(args, "edge_tanh_temperature", 1.0),
                edge_ig_reward_lambda=edge_ig_reward_lambda,
                edge_ig_discount_factor=getattr(args, "edge_ig_discount_factor", 0.0),
                advantage_epsilon=getattr(args, "graph_advantage_epsilon", 1e-6),
                graph_ib_beta=getattr(args, "graph_ib_beta", 0.2),
                graph_ib_prior_prob=getattr(args, "graph_ib_prior_prob", 0.45),
            )
            if graph_tf_corrects:
                avg_correct = sum(graph_tf_corrects) / len(graph_tf_corrects)
                avg_edges = sum(graph_tf_edge_counts) / len(graph_tf_edge_counts)
                avg_adv_variance = (
                    sum(summary["correctness_variance"] for summary in tf_summaries)
                    / len(tf_summaries)
                    if tf_summaries
                    else 0.0
                )
                avg_adv_std = (
                    sum(summary["correctness_std"] for summary in tf_summaries)
                    / len(tf_summaries)
                    if tf_summaries
                    else 0.0
                )
                print(
                    "graph correctness metrics: "
                    f"accuracy={avg_correct:.6f}, "
                    f"avg_edges={avg_edges:.6f}, "
                    f"avg_adv_variance={avg_adv_variance:.6f}, "
                    f"avg_adv_std={avg_adv_std:.6f}, "
                    f"num_graphs={len(graph_tf_corrects)}"
                )
        else:
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
                    use_semantic_edges
                    and (is_solved or edge_ig_reward_lambda != 0.0)
                    and bool(realized_graph.edge_log_probs)
                ):
                    edge_rewards, edge_details = await edge_entropy_rewards(
                        realized_graph,
                        record["task"],
                        input_dict,
                        semantic_judge,
                        effective_num_entropy_samples,
                        negative_reward_scale=args.negative_edge_reward_scale,
                        nonpositive_penalty=args.nonpositive_edge_penalty,
                        kle_heat_t=getattr(args, "kle_heat_t", 0.3),
                        target_spec=make_target_spec(dataset_name, true_answer),
                        ig_scorer=tf_scorer,
                    )
                    if selector_buffer is not None and is_solved:
                        selector_buffer.add_many(build_edge_selector_examples(
                            realized_graph,
                            record["task"],
                            edge_details,
                            getattr(args, "selector_ig_tau", 0.0),
                        ))
                realized_graph.clear_execution_history()
                utility = {
                    "correctness": correctness_reward,
                    "edge_entropy_rewards": edge_rewards,
                }
                utilities.append(utility)
                single_loss = -log_prob * float(correctness_reward)
                if edge_ig_reward_lambda != 0.0:
                    edge_ig_loss, edge_ig_summary = edge_information_gain_loss(
                        realized_graph,
                        edge_details,
                        log_prob,
                        edge_tanh_temperature=getattr(args, "edge_tanh_temperature", 1.0),
                        edge_ig_reward_lambda=edge_ig_reward_lambda,
                        edge_ig_discount_factor=getattr(args, "edge_ig_discount_factor", 0.0),
                    )
                    single_loss = single_loss + edge_ig_loss
                    utility["edge_ig_loss_summary"] = edge_ig_summary
                loss_list.append(single_loss)

            utility_loss = torch.mean(torch.stack(loss_list))
        total_loss = utility_loss
        if train_updates_enabled:
            optimizer.zero_grad()
            if total_loss.requires_grad:
                total_loss.backward()
                optimizer.step()
            else:
                print("Skipping graph optimizer step: no differentiable edge decisions.")
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
        if not use_multi_graph_reward:
            print("utilities:", utilities)
        print("utility loss:", utility_loss.item())
        print("loss:", total_loss.item())

        if i_batch + 1 == args.num_iterations:
            save_graph_checkpoint(
                graph,
                getattr(args, "checkpoint_file", f"result/checkpoints/{dataset_name}.pt"),
                dataset=dataset_name,
                args=args,
                optimizer=optimizer,
                edge_selector=edge_selector if selector_trained else None,
                metrics={"train_accuracy": accuracy},
            )
            total_solved = 0
            total_executed = 0
            total_edges = 0
            edge_samples = 0
            accuracy = 0.0
            graph.gat.eval()
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
