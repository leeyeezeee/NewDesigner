import asyncio
import copy
import random
import time
from typing import Callable, List, Literal, Union

import torch

from GDesigner.graph.graph import Graph
from GDesigner.utils.globals import CompletionTokens, Cost, PromptTokens, Time
from GDesigner.utils.metrics import (
    reset_usage_counters,
    usage_delta,
    usage_snapshot,
    write_metrics_record,
)
from experiments.agent_backend import apply_agent_backend_args
from GDesigner.utils.edge_selector import (
    EdgeSelector,
    SelectorReplayBuffer,
    build_edge_selector_examples,
    train_edge_selector,
)
from GDesigner.utils.ig_rewards import (
    compute_edge_information_gain,
)
from GDesigner.utils.ig_scorer import FinalAnswerScorer, make_target_spec
from experiments.checkpoint import save_graph_checkpoint
from experiments.graph_concurrency import limited_graph_arun, make_graph_semaphore
from experiments.graph_critic import (
    build_graph_critic,
    critic_counterfactual_edge_loss,
    score_full_graph_teacher_forcing,
    train_graph_critic,
)
from experiments.refinement_loss import refinement_regularization_loss
from experiments.teacher_forcing_reward import (
    edge_information_gain_loss,
    experiment_summary_metadata,
    graph_correctness_advantage_edge_loss,
    set_experiment_seed,
)
from experiments.edge_training_log import (
    append_case_record,
    append_training_step,
    reset_case_log,
    reset_edge_training_log,
    resolve_case_log_file,
    resolve_edge_training_log_file,
    resolve_question_id,
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
    set_experiment_seed(args.seed)
    apply_agent_backend_args(args)
    reset_usage_counters()
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
        refine_rank=args.refine_rank,
        **kwargs,
    )
    graph.gat.train()
    graph.edge_mlp.train()
    graph.spatial_affinity.train()
    edge_training_log_file = resolve_edge_training_log_file(
        dataset_name,
    )
    reset_edge_training_log(edge_training_log_file)
    case_log_file = resolve_case_log_file(dataset_name)
    reset_case_log(case_log_file)
    optimizer_params = graph.spatial_parameters()
    if graph.optimized_temporal:
        optimizer_params.append(graph.temporal_logits)
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)

    use_graph_tf_reward = bool(getattr(args, "use_graph_tf_reward", False))
    raw_edge_ig_reward_lambda = getattr(args, "edge_ig_reward_lambda", None)
    edge_ig_reward_lambda = (
        0.0
        if raw_edge_ig_reward_lambda is None
        else float(raw_edge_ig_reward_lambda)
    )
    use_graph_critic = bool(getattr(args, "use_graph_critic", False))
    use_multi_graph_reward = use_graph_tf_reward or use_graph_critic
    optimize_enabled = args.optimized_spatial or args.optimized_temporal
    if use_graph_critic and not args.optimized_spatial:
        raise ValueError("--use_graph_critic requires --optimized_spatial.")
    if use_graph_critic and int(getattr(args, "graph_sample_count", 8)) < 2:
        raise ValueError("--use_graph_critic requires --graph_sample_count >= 2.")
    graph_critic = None
    graph_critic_optimizer = None
    if use_graph_critic:
        graph_critic, graph_critic_optimizer = build_graph_critic(
            graph, learning_rate=getattr(args, "graph_critic_lr", 1e-3)
        )
    record_edge_ig = (
        optimize_enabled
        and (
            args.use_edge_selector
            or edge_ig_reward_lambda != 0.0
            or (bool(edge_training_log_file) and not use_graph_critic)
        )
    )
    edge_selector = None
    selector_buffer = None
    selector_optimizer = None
    selector_trained = False
    if args.use_edge_selector and record_edge_ig:
        edge_selector = EdgeSelector(graph.features.size(1))
        selector_buffer = SelectorReplayBuffer(getattr(args, "selector_buffer_size", 512))
        selector_optimizer = torch.optim.Adam(edge_selector.parameters(), lr=1e-3)
    tf_scorer = (
        FinalAnswerScorer()
        if (
            args.use_edge_selector
            or edge_ig_reward_lambda != 0.0
            or bool(edge_training_log_file)
            or use_graph_critic
        )
        else None
    )
    graph_semaphore = make_graph_semaphore(getattr(args, "max_concurrent_graphs", 10))

    num_batches = int(len(dataset) / args.batch_size)
    total_solved, total_executed = (0, 0)
    total_edges, edge_samples = (0, 0)
    accuracy = 0.0
    train_usage = {"cost": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "llm_calls": 0.0}
    train_wall_start = time.time()

    for i_batch in range(num_batches):
        train_updates_enabled = optimize_enabled and i_batch < args.num_iterations
        edge_ig_measurement_enabled = (
            train_updates_enabled
            and i_batch >= max(0, int(args.edge_ig_warmup_iterations))
        )
        iteration_edge_ig_reward_lambda = (
            edge_ig_reward_lambda if edge_ig_measurement_enabled else 0.0
        )
        record_edge_ig_history = record_edge_ig and train_updates_enabled
        batch_edge_selector = edge_selector if (selector_trained and not train_updates_enabled) else None
        print(f"Batch {i_batch}", 80 * "-")
        start_ts = time.time()
        answer_log_probs = []
        answers = []
        realized_graphs = []
        input_dicts = []
        record_input_dicts = []
        question_ids = []
        sample_groups = []
        batch_correctness: List[float] = []
        rollout_usage_before = usage_snapshot()

        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if current_batch is None:
            print("No more data available.")
            break

        for i_record, record in enumerate(current_batch):
            question_ids.append(resolve_question_id(
                record, i_batch * args.batch_size + i_record
            ))
            task = record["task"]
            answers.append(record["answer"])
            input_dict = {"task": task}
            record_input_dicts.append(input_dict)
            group_indices = []
            sample_count = (
                max(1, int(getattr(args, "graph_sample_count", 8)))
                if (use_multi_graph_reward and train_updates_enabled)
                else 1
            )
            for _ in range(sample_count):
                realized_graph = copy.deepcopy(graph)
                realized_graph.gat = graph.gat
                realized_graph.edge_mlp = graph.edge_mlp
                realized_graph.spatial_affinity = graph.spatial_affinity
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
                            record_execution_history=(record_edge_ig_history or not train_updates_enabled),
                            track_grad=train_updates_enabled,
                            edge_selector=batch_edge_selector,
                        )
                    )
                )
            sample_groups.append(group_indices)

        raw_results = await asyncio.gather(*answer_log_probs)
        rollout_usage = usage_delta(rollout_usage_before, usage_snapshot())
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[dict] = []

        if use_multi_graph_reward and train_updates_enabled:
            graph_groups = []
            graph_log_prob_groups = []
            correctness_groups = []
            edge_detail_groups = []
            graph_critic_score_groups = []
            graph_tf_corrects: List[float] = []
            graph_tf_edge_counts: List[float] = []
            for record, true_answer, group_indices, input_dict, question_id in zip(
                current_batch,
                answers,
                sample_groups,
                record_input_dicts,
                question_ids,
            ):
                target_spec = make_target_spec(dataset_name, true_answer)
                graph_group = []
                graph_log_prob_group = []
                correctness_group = []
                edge_detail_group = []
                graph_critic_score_group = []
                for sample_pos, graph_idx in enumerate(group_indices):
                    realized_graph = realized_graphs[graph_idx]
                    answer = raw_answers[graph_idx]
                    predict_answer = answer_parser(answer[0])
                    is_solved = correctness_fn(predict_answer, true_answer)
                    correctness_reward = float(is_solved)
                    batch_correctness.append(correctness_reward)
                    graph_tf_corrects.append(correctness_reward)
                    graph_tf_edge_counts.append(realized_graph.mean_spatial_edges_per_round)
                    needs_edge_details = (
                        record_edge_ig_history
                        and edge_ig_measurement_enabled
                        and bool(realized_graph.edge_log_probs)
                        and (
                            iteration_edge_ig_reward_lambda != 0.0
                            or selector_buffer is not None
                            or bool(edge_training_log_file)
                        )
                    )
                    if needs_edge_details:
                        edge_details = await compute_edge_information_gain(
                            realized_graph,
                            input_dict,
                            target_spec=target_spec,
                            scorer=tf_scorer,
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
                    if use_graph_critic:
                        graph_critic_score_group.append(
                            await score_full_graph_teacher_forcing(
                                realized_graph,
                                input_dict,
                                target_spec=target_spec,
                                scorer=tf_scorer,
                            )
                        )
                    realized_graph.clear_execution_history()
                graph_groups.append(graph_group)
                graph_log_prob_groups.append(graph_log_prob_group)
                correctness_groups.append(correctness_group)
                edge_detail_groups.append(edge_detail_group)
                graph_critic_score_groups.append(graph_critic_score_group)
            reference_loss = torch.mean(torch.stack(list(log_probs)))
            utility_loss, tf_summaries = graph_correctness_advantage_edge_loss(
                graph_groups,
                graph_log_prob_groups,
                correctness_groups,
                edge_detail_groups,
                reference_loss,
                edge_tanh_temperature=getattr(args, "edge_tanh_temperature", 1.0),
                edge_ig_reward_lambda=iteration_edge_ig_reward_lambda,
                edge_ig_discount_factor=getattr(args, "edge_ig_discount_factor", 0.0),
                advantage_epsilon=getattr(args, "graph_advantage_epsilon", 1e-6),
            )
            critic_reward_summary = None
            if (
                use_graph_critic
                and i_batch >= max(
                    0, int(getattr(args, "graph_critic_warmup_iterations", 2))
                )
            ):
                critic_reward_loss, critic_reward_summary = (
                    critic_counterfactual_edge_loss(
                        graph_critic,
                        graph_groups,
                        record_input_dicts,
                        reference_loss,
                        reward_lambda=getattr(
                            args, "graph_critic_reward_lambda", 0.2
                        ),
                        tanh_temperature=getattr(
                            args, "edge_tanh_temperature", 1.0
                        ),
                    )
                )
                utility_loss = utility_loss + critic_reward_loss
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
                    "graph reward metrics: "
                    f"accuracy={avg_correct:.6f}, "
                    f"avg_edges={avg_edges:.6f}, "
                    f"avg_adv_variance={avg_adv_variance:.6f}, "
                    f"avg_adv_std={avg_adv_std:.6f}, "
                    f"num_graphs={len(graph_tf_corrects)}"
                )
        else:
            for graph_idx, (record, answer, log_prob, true_answer, realized_graph, input_dict, question_id) in enumerate(zip(
                current_batch,
                raw_answers,
                log_probs,
                answers,
                realized_graphs,
                input_dicts,
                question_ids,
            )):
                predict_answer = answer_parser(answer[0])
                is_solved = correctness_fn(predict_answer, true_answer)
                correctness_reward = float(is_solved)
                batch_correctness.append(correctness_reward)
                total_solved += int(is_solved)
                total_executed += 1
                accuracy = total_solved / total_executed
                if not train_updates_enabled:
                    total_edges += realized_graph.mean_spatial_edges_per_round
                    edge_samples += 1

                edge_details = {}
                if (
                    record_edge_ig_history
                    and edge_ig_measurement_enabled
                    and (
                        is_solved
                        or iteration_edge_ig_reward_lambda != 0.0
                        or bool(edge_training_log_file)
                    )
                    and bool(realized_graph.edge_log_probs)
                ):
                    edge_details = await compute_edge_information_gain(
                        realized_graph,
                        input_dict,
                        target_spec=make_target_spec(dataset_name, true_answer),
                        scorer=tf_scorer,
                    )
                    if selector_buffer is not None and is_solved:
                        selector_buffer.add_many(build_edge_selector_examples(
                            realized_graph,
                            record["task"],
                            edge_details,
                            getattr(args, "selector_ig_tau", 0.0),
                        ))
                if not train_updates_enabled:
                    append_case_record(
                        case_log_file,
                        question_id=question_id,
                        question=input_dict["task"],
                        graph=realized_graph,
                        final_answer=predict_answer,
                        correct=is_solved,
                    )
                realized_graph.clear_execution_history()
                utility = {
                    "correctness": correctness_reward,
                }
                utilities.append(utility)
                single_loss = -log_prob * float(correctness_reward)
                if iteration_edge_ig_reward_lambda != 0.0:
                    edge_ig_loss, edge_ig_summary = edge_information_gain_loss(
                        realized_graph,
                        edge_details,
                        log_prob,
                        edge_tanh_temperature=getattr(args, "edge_tanh_temperature", 1.0),
                        edge_ig_reward_lambda=iteration_edge_ig_reward_lambda,
                        edge_ig_discount_factor=getattr(args, "edge_ig_discount_factor", 0.0),
                    )
                    single_loss = single_loss + edge_ig_loss
                    utility["edge_ig_loss_summary"] = edge_ig_summary
                loss_list.append(single_loss)

            utility_loss = torch.mean(torch.stack(loss_list))
        if train_updates_enabled:
            reg_loss, anchor_loss, sparse_loss = refinement_regularization_loss(
                realized_graphs,
                utility_loss,
                anchor_reg_weight=args.anchor_reg_weight,
                sparsity_reg_weight=args.sparsity_reg_weight,
            )
        else:
            reg_loss = utility_loss.new_tensor(0.0)
            anchor_loss = utility_loss.new_tensor(0.0)
            sparse_loss = utility_loss.new_tensor(0.0)
        total_loss = utility_loss + reg_loss
        if train_updates_enabled:
            append_training_step(
                edge_training_log_file,
                step=i_batch,
                accuracy=sum(batch_correctness) / len(batch_correctness),
                avg_edges=(
                    sum(item.mean_spatial_edges_per_round for item in realized_graphs)
                    / len(realized_graphs)
                ),
                avg_communication_tokens=(
                    rollout_usage["prompt_tokens"] + rollout_usage["completion_tokens"]
                ) / len(realized_graphs),
            )
        if train_updates_enabled:
            optimizer.zero_grad()
            if not total_loss.requires_grad:
                raise RuntimeError(
                    "Graph training loss is not differentiable. A zero-edge sample "
                    "must still retain policy or refinement gradients."
                )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Graph training loss is non-finite: {total_loss.detach().item()}."
                )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                optimizer_params,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if use_graph_critic:
                critic_fit_summary = train_graph_critic(
                    graph_critic,
                    graph_critic_optimizer,
                    graph_groups,
                    record_input_dicts,
                    graph_critic_score_groups,
                )
                print(
                    "graph critic: "
                    f"mse={critic_fit_summary['loss']:.6f}, "
                    f"target_std={critic_fit_summary['target_std']:.6f}, "
                    "predicted_edges="
                    f"{int((critic_reward_summary or {}).get('edge_count', 0))}"
                )
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
        print("anchor loss:", anchor_loss.item())
        print("nuclear sparsity loss:", sparse_loss.item())
        print("loss:", total_loss.item())

        if i_batch + 1 == args.num_iterations:
            train_usage = usage_snapshot()
            train_wall_seconds = time.time() - train_wall_start
            save_graph_checkpoint(
                graph,
                getattr(args, "checkpoint_file", f"result/checkpoints/{dataset_name}.pt"),
                dataset=dataset_name,
                args=args,
                optimizer=optimizer,
                edge_selector=edge_selector if selector_trained else None,
                graph_critic=graph_critic,
                graph_critic_optimizer=graph_critic_optimizer,
                metrics={"train_accuracy": accuracy},
            )
            total_solved = 0
            total_executed = 0
            total_edges = 0
            edge_samples = 0
            accuracy = 0.0
            graph.gat.eval()
            graph.edge_mlp.eval()
            graph.spatial_affinity.eval()
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
    summary_record = {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "total_solved": total_solved,
        "total_executed": total_executed,
        "avg_edges": avg_edges,
        "llm_name": args.llm_name,
        **experiment_summary_metadata(args, dataset_name),
        "train_llm_calls": int(train_usage["llm_calls"]),
        "train_prompt_tokens": int(train_usage["prompt_tokens"]),
        "train_completion_tokens": int(train_usage["completion_tokens"]),
        "train_cost": float(train_usage["cost"]),
        "train_wall_seconds": float(locals().get("train_wall_seconds", 0.0)),
    }
    write_metrics_record(args.metrics_file, summary_record)


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
