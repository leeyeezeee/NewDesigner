import torch
import json
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy
from pathlib import Path

from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from GDesigner.utils.metrics import usage_delta, usage_snapshot
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
    graph_correctness_advantage_edge_loss,
)
from experiments.edge_training_log import (
    append_training_step,
    reset_edge_training_log,
    resolve_edge_training_log_file,
    resolve_question_id,
)

_GRAPH_TF_RECORD_FILE = "mmlu_graph_tf_records.jsonl"


def _reset_jsonl(path: str) -> None:
    record_path = Path(path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("", encoding="utf-8")


def _append_jsonl(path: str, record: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


async def train(graph:Graph,
            dataset,
            num_iters:int=100,
            num_rounds:int=1,
            lr:float=0.001,
            batch_size:int = 4,
            use_edge_selector: bool = False,
            imp_per_iterations: int = 5,
            pruning_rate: float = 0.25,
            selector_buffer_size: int = 512,
            selector_ig_tau: float = 0.0,
            use_graph_tf_reward: bool = False,
            use_graph_critic: bool = False,
            graph_sample_count: int = 8,
            graph_critic_lr: float = 1e-3,
            graph_critic_reward_lambda: float = 0.2,
            graph_critic_warmup_iterations: int = 2,
            edge_tanh_temperature: float = 1.0,
            edge_ig_reward_lambda: float = None,
            edge_ig_warmup_iterations: int = 2,
            edge_ig_discount_factor: float = 0.0,
            graph_advantage_epsilon: float = 1e-6,
            max_concurrent_graphs: int = 10,
            anchor_reg_weight: float = 0.0,
            sparsity_reg_weight: float = 0.0,
          ):
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record
    
    loader = infinite_data_loader()
    resolved_edge_ig_reward_lambda = (
        0.0
        if edge_ig_reward_lambda is None
        else float(edge_ig_reward_lambda)
    )
    use_multi_graph_reward = use_graph_tf_reward or use_graph_critic
    if use_graph_critic and not graph.optimized_spatial:
        raise ValueError("--use_graph_critic requires --optimized_spatial.")
    if use_graph_critic and int(graph_sample_count) < 2:
        raise ValueError("--use_graph_critic requires --graph_sample_count >= 2.")
    graph_critic = None
    graph_critic_optimizer = None
    if use_graph_critic:
        graph_critic, graph_critic_optimizer = build_graph_critic(
            graph,
            learning_rate=graph_critic_lr,
        )
    # Real edge ablation remains the default.  The critic flag switches this
    # diagnostic-only default off, while explicit selector/IG requests still
    # opt back into the unchanged real-ablation implementation.
    record_edge_training = not use_graph_critic
    edge_training_log_path = resolve_edge_training_log_file("mmlu")
    reset_edge_training_log(edge_training_log_path)
    record_edge_ig = (
        use_edge_selector
        or resolved_edge_ig_reward_lambda != 0.0
        or record_edge_training
    )
    edge_selector = None
    selector_buffer = None
    selector_optimizer = None
    selector_trained = False
    if use_edge_selector and record_edge_ig:
        edge_selector = EdgeSelector(graph.features.size(1))
        selector_buffer = SelectorReplayBuffer(selector_buffer_size)
        selector_optimizer = torch.optim.Adam(edge_selector.parameters(), lr=1e-3)
    tf_scorer = (
        FinalAnswerScorer()
        if (
            use_edge_selector
            or resolved_edge_ig_reward_lambda != 0.0
            or record_edge_training
            or use_graph_critic
        )
        else None
    )
    graph_semaphore = make_graph_semaphore(max_concurrent_graphs)
    
    optimizer_params = graph.spatial_parameters()
    if graph.optimized_temporal:
        optimizer_params.append(graph.temporal_logits)
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    graph.gat.train()
    graph.edge_mlp.train()
    graph.spatial_affinity.train()
    for i_iter in range(num_iters):
        edge_ig_measurement_enabled = i_iter >= max(
            0, int(edge_ig_warmup_iterations)
        )
        iteration_edge_ig_reward_lambda = (
            resolved_edge_ig_reward_lambda
            if edge_ig_measurement_enabled
            else 0.0
        )
        print(f"Iter {i_iter}", 80*'-')
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []
        realized_graphs = []
        input_dicts = []
        record_input_dicts = []
        question_ids = []
        sample_groups = []
        batch_correctness: List[float] = []
        rollout_usage_before = usage_snapshot()

        for i_record, record in zip(range(batch_size), loader):
            question_ids.append(resolve_question_id(
                record, i_iter * batch_size + i_record
            ))
            input_dict = dataset.record_to_input(record)
            record_input_dicts.append(input_dict)
            group_indices = []
            sample_count = max(1, int(graph_sample_count)) if use_multi_graph_reward else 1
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
                answer_log_probs.append(asyncio.create_task(
                    limited_graph_arun(
                        graph_semaphore,
                        realized_graph,
                        input_dict,
                        num_rounds,
                        record_execution_history=record_edge_ig,
                        track_grad=True,
                    )
                ))
            sample_groups.append(group_indices)
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)
        
        raw_results = await asyncio.gather(*answer_log_probs)
        rollout_usage = usage_delta(rollout_usage_before, usage_snapshot())
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[dict] = []
        graph_tf_corrects: List[float] = []
        graph_tf_edge_counts: List[float] = []
        answers: List[str] = []
        
        if use_multi_graph_reward:
            graph_groups = []
            graph_log_prob_groups = []
            correctness_groups = []
            edge_detail_groups = []
            graph_critic_score_groups = []
            for record_idx, (correct_answer, group_indices, input_dict, question_id) in enumerate(zip(
                correct_answers,
                sample_groups,
                record_input_dicts,
                question_ids,
            )):
                assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                target_spec = make_target_spec("mmlu", correct_answer)
                graph_group = []
                graph_log_prob_group = []
                correctness_group = []
                edge_detail_group = []
                graph_critic_score_group = []
                for sample_pos, graph_idx in enumerate(group_indices):
                    raw_answer = raw_answers[graph_idx]
                    realized_graph = realized_graphs[graph_idx]
                    record_answer = dataset.postprocess_answer(raw_answer)
                    record_accuracy = Accuracy()
                    record_accuracy.update(record_answer, correct_answer)
                    graph_tf_corrects.append(float(record_accuracy.get()))
                    batch_correctness.append(float(record_accuracy.get()))
                    graph_tf_edge_counts.append(realized_graph.mean_spatial_edges_per_round)
                    needs_edge_details = (
                        edge_ig_measurement_enabled
                        and bool(realized_graph.edge_log_probs)
                        and (
                            iteration_edge_ig_reward_lambda != 0.0
                            or selector_buffer is not None
                            or record_edge_training
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
                            input_dict["task"],
                            edge_details,
                            selector_ig_tau,
                        ))
                    if sample_pos == 0:
                        answer = dataset.postprocess_answer(raw_answer)
                        answers.append(answer)
                        accuracy = Accuracy()
                        accuracy.update(answer, correct_answer)
                        correctness_reward = accuracy.get()
                        utilities.append({
                            "correctness": correctness_reward,
                        })
                    graph_group.append(realized_graph)
                    graph_log_prob_group.append(log_probs[graph_idx])
                    correctness_group.append(float(record_accuracy.get()))
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
                edge_tanh_temperature=edge_tanh_temperature,
                edge_ig_reward_lambda=iteration_edge_ig_reward_lambda,
                edge_ig_discount_factor=edge_ig_discount_factor,
                advantage_epsilon=graph_advantage_epsilon,
            )
            critic_reward_summary = None
            if (
                use_graph_critic
                and i_iter >= max(0, int(graph_critic_warmup_iterations))
            ):
                critic_reward_loss, critic_reward_summary = (
                    critic_counterfactual_edge_loss(
                        graph_critic,
                        graph_groups,
                        record_input_dicts,
                        reference_loss,
                        reward_lambda=graph_critic_reward_lambda,
                        tanh_temperature=edge_tanh_temperature,
                    )
                )
                utility_loss = utility_loss + critic_reward_loss
            if graph_tf_corrects:
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
                graph_tf_summary = {
                    "iteration": i_iter,
                    "num_graphs": len(graph_tf_corrects),
                    "accuracy": sum(graph_tf_corrects) / len(graph_tf_corrects),
                    "avg_edges": sum(graph_tf_edge_counts) / len(graph_tf_edge_counts),
                    "avg_adv_variance": avg_adv_variance,
                    "avg_adv_std": avg_adv_std,
                }
                print(
                    "graph reward metrics: "
                    f"accuracy={graph_tf_summary['accuracy']:.6f}, "
                    f"avg_edges={graph_tf_summary['avg_edges']:.6f}, "
                    f"avg_adv_variance={avg_adv_variance:.6f}, "
                    f"avg_adv_std={avg_adv_std:.6f}, "
                    f"num_graphs={graph_tf_summary['num_graphs']}"
                )
        else:
            for graph_idx, (raw_answer, log_prob, correct_answer, realized_graph, input_dict, question_id) in enumerate(zip(raw_answers, log_probs, correct_answers, realized_graphs, input_dicts, question_ids)):
                answer = dataset.postprocess_answer(raw_answer)
                answers.append(answer)
                assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                correctness_reward = accuracy.get()
                batch_correctness.append(float(correctness_reward))
                edge_details = {}
                if (
                    record_edge_ig
                    and edge_ig_measurement_enabled
                    and (
                        correctness_reward > 0
                        or iteration_edge_ig_reward_lambda != 0.0
                        or record_edge_training
                    )
                    and bool(realized_graph.edge_log_probs)
                ):
                    edge_details = await compute_edge_information_gain(
                        realized_graph,
                        input_dict,
                        target_spec=make_target_spec("mmlu", correct_answer),
                        scorer=tf_scorer,
                    )
                    if selector_buffer is not None and correctness_reward > 0:
                        selector_buffer.add_many(build_edge_selector_examples(
                            realized_graph,
                            input_dict["task"],
                            edge_details,
                            selector_ig_tau,
                        ))
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
                        edge_tanh_temperature=edge_tanh_temperature,
                        edge_ig_reward_lambda=iteration_edge_ig_reward_lambda,
                        edge_ig_discount_factor=edge_ig_discount_factor,
                    )
                    single_loss = single_loss + edge_ig_loss
                    utility["edge_ig_loss_summary"] = edge_ig_summary
                loss_list.append(single_loss)
                print(f"correct answer:{correct_answer}")

            utility_loss = torch.mean(torch.stack(loss_list))
        reg_loss, anchor_loss, sparse_loss = refinement_regularization_loss(
            realized_graphs,
            utility_loss,
            anchor_reg_weight=anchor_reg_weight,
            sparsity_reg_weight=sparsity_reg_weight,
        )
        total_loss = utility_loss + reg_loss
        append_training_step(
            edge_training_log_path,
            step=i_iter,
            accuracy=sum(batch_correctness) / len(batch_correctness),
            avg_edges=(
                sum(item.mean_spatial_edges_per_round for item in realized_graphs)
                / len(realized_graphs)
            ),
            avg_communication_tokens=(
                rollout_usage["prompt_tokens"] + rollout_usage["completion_tokens"]
            ) / len(realized_graphs),
        )
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
            and (i_iter + 1) % imp_per_iterations == 0
        ):
            temporal_masks, pruned_temporal_idx = graph.prune_temporal_edges(pruning_rate)
            print(f"pruned temporal edges: {pruned_temporal_idx.numel()}")
            print("temporal masks:", temporal_masks.view(graph.num_nodes, graph.num_nodes))

        print("answers:",answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        if not use_multi_graph_reward:
            print("utilities:", utilities) # [0.0, 0.0, 0.0, 1.0]
        print("utility loss:", utility_loss.item())
        print("anchor loss:", anchor_loss.item())
        print("nuclear sparsity loss:", sparse_loss.item())
        print("loss:", total_loss.item()) # 4.6237263679504395
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    if use_graph_critic:
        graph.graph_critic = graph_critic
        graph.graph_critic_optimizer = graph_critic_optimizer
    return edge_selector if selector_trained else None
        
