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
from experiments.graph_concurrency import limited_graph_arun, make_graph_semaphore
from experiments.teacher_forcing_reward import (
    edge_information_gain_loss,
    graph_correctness_advantage_edge_loss,
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
            lr:float=0.1,
            batch_size:int = 4,
            use_edge_selector: bool = False,
            imp_per_iterations: int = 5,
            pruning_rate: float = 0.25,
            num_entropy_samples: int = 1,
            kle_heat_t: float = 0.3,
            semantic_judge_llm_name: str = "gpt-4o-mini",
            semantic_judge_api_key: str = "",
            semantic_judge_base_url: str = "",
            semantic_judge_model_path: str = "",
            semantic_judge_max_concurrency: int = None,
            negative_edge_reward_scale: float = 1.0,
            nonpositive_edge_penalty: float = 0.01,
            selector_buffer_size: int = 512,
            selector_ig_tau: float = 0.0,
            use_graph_tf_reward: bool = False,
            use_graph_correctness_advantage: bool = False,
            graph_sample_count: int = 5,
            graph_softmax_temperature: float = 1.0,
            edge_tanh_temperature: float = 1.0,
            edge_ig_reward_lambda: float = None,
            edge_ig_discount_factor: float = 0.0,
            graph_advantage_epsilon: float = 1e-6,
            graph_ib_beta: float = 1.0,
            graph_ib_prior_prob: float = 0.45,
            max_concurrent_graphs: int = 10,
          ):
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record
    
    loader = infinite_data_loader()
    effective_num_entropy_samples = 1
    resolved_edge_ig_reward_lambda = (
        0.0
        if edge_ig_reward_lambda is None
        else float(edge_ig_reward_lambda)
    )
    use_multi_graph_reward = use_graph_tf_reward or use_graph_correctness_advantage
    use_semantic_edges = (
        use_edge_selector
        or resolved_edge_ig_reward_lambda != 0.0
    )
    batch_entropy_samples = 1
    semantic_judge = None
    edge_selector = None
    selector_buffer = None
    selector_optimizer = None
    selector_trained = False
    if use_edge_selector and use_semantic_edges:
        edge_selector = EdgeSelector(graph.features.size(1))
        selector_buffer = SelectorReplayBuffer(selector_buffer_size)
        selector_optimizer = torch.optim.Adam(edge_selector.parameters(), lr=1e-3)
    tf_scorer = (
        FinalAnswerScorer()
        if (use_edge_selector or resolved_edge_ig_reward_lambda != 0.0)
        else None
    )
    if use_multi_graph_reward:
        _reset_jsonl(_GRAPH_TF_RECORD_FILE)
    graph_semaphore = make_graph_semaphore(max_concurrent_graphs)
    
    optimizer_params = list(graph.gat.parameters()) + graph.refinement_parameters()
    if graph.optimized_temporal:
        optimizer_params.append(graph.temporal_logits)
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    graph.gat.train()
    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80*'-')
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []
        realized_graphs = []
        input_dicts = []
        record_input_dicts = []
        sample_groups = []

        for i_record, record in zip(range(batch_size), loader):
            input_dict = dataset.record_to_input(record)
            record_input_dicts.append(input_dict)
            group_indices = []
            sample_count = max(1, int(graph_sample_count)) if use_multi_graph_reward else 1
            for _ in range(sample_count):
                realized_graph = copy.deepcopy(graph)
                realized_graph.gat = graph.gat
                realized_graph.spatial_affinity_weight = graph.spatial_affinity_weight
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
                        num_entropy_samples=batch_entropy_samples,
                        record_execution_history=use_semantic_edges,
                        track_grad=True,
                    )
                ))
            sample_groups.append(group_indices)
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)
        
        raw_results = await asyncio.gather(*answer_log_probs)
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
            for record_idx, (correct_answer, group_indices, input_dict) in enumerate(zip(
                correct_answers,
                sample_groups,
                record_input_dicts,
            )):
                assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                target_spec = make_target_spec("mmlu", correct_answer)
                graph_group = []
                graph_log_prob_group = []
                correctness_group = []
                edge_detail_group = []
                for sample_pos, graph_idx in enumerate(group_indices):
                    raw_answer = raw_answers[graph_idx]
                    realized_graph = realized_graphs[graph_idx]
                    record_answer = dataset.postprocess_answer(raw_answer)
                    record_accuracy = Accuracy()
                    record_accuracy.update(record_answer, correct_answer)
                    graph_tf_corrects.append(float(record_accuracy.get()))
                    graph_tf_edge_counts.append(float(sum(realized_graph.realized_edge_counts)))
                    needs_edge_details = (
                        bool(realized_graph.edge_log_probs)
                        and (
                            resolved_edge_ig_reward_lambda != 0.0
                            or selector_buffer is not None
                        )
                    )
                    if needs_edge_details:
                        _edge_rewards, edge_details = await edge_entropy_rewards(
                            realized_graph,
                            input_dict["task"],
                            input_dict,
                            semantic_judge,
                            effective_num_entropy_samples,
                            negative_reward_scale=negative_edge_reward_scale,
                            nonpositive_penalty=nonpositive_edge_penalty,
                            kle_heat_t=kle_heat_t,
                            target_spec=target_spec,
                            ig_scorer=tf_scorer,
                            compute_rewards=False,
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
                edge_tanh_temperature=edge_tanh_temperature,
                edge_ig_reward_lambda=resolved_edge_ig_reward_lambda,
                edge_ig_discount_factor=edge_ig_discount_factor,
                advantage_epsilon=graph_advantage_epsilon,
                graph_ib_beta=graph_ib_beta,
                graph_ib_prior_prob=graph_ib_prior_prob,
            )
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
                _append_jsonl(_GRAPH_TF_RECORD_FILE, graph_tf_summary)
                print(
                    "graph correctness metrics: "
                    f"accuracy={graph_tf_summary['accuracy']:.6f}, "
                    f"avg_edges={graph_tf_summary['avg_edges']:.6f}, "
                    f"avg_adv_variance={avg_adv_variance:.6f}, "
                    f"avg_adv_std={avg_adv_std:.6f}, "
                    f"num_graphs={graph_tf_summary['num_graphs']}"
                )
        else:
            for raw_answer, log_prob, correct_answer, realized_graph, input_dict in zip(raw_answers, log_probs, correct_answers, realized_graphs, input_dicts):
                answer = dataset.postprocess_answer(raw_answer)
                answers.append(answer)
                assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                correctness_reward = accuracy.get()
                edge_rewards = {}
                edge_details = {}
                if (
                    use_semantic_edges
                    and (correctness_reward > 0 or resolved_edge_ig_reward_lambda != 0.0)
                    and bool(realized_graph.edge_log_probs)
                ):
                    edge_rewards, edge_details = await edge_entropy_rewards(
                        realized_graph,
                        input_dict["task"],
                        input_dict,
                        semantic_judge,
                        effective_num_entropy_samples,
                        negative_reward_scale=negative_edge_reward_scale,
                        nonpositive_penalty=nonpositive_edge_penalty,
                        kle_heat_t=kle_heat_t,
                        target_spec=make_target_spec("mmlu", correct_answer),
                        ig_scorer=tf_scorer,
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
                    "edge_entropy_rewards": edge_rewards,
                }
                utilities.append(utility)
                single_loss = -log_prob * float(correctness_reward)
                if resolved_edge_ig_reward_lambda != 0.0:
                    edge_ig_loss, edge_ig_summary = edge_information_gain_loss(
                        realized_graph,
                        edge_details,
                        log_prob,
                        edge_tanh_temperature=edge_tanh_temperature,
                        edge_ig_reward_lambda=resolved_edge_ig_reward_lambda,
                        edge_ig_discount_factor=edge_ig_discount_factor,
                    )
                    single_loss = single_loss + edge_ig_loss
                    utility["edge_ig_loss_summary"] = edge_ig_summary
                loss_list.append(single_loss)
                print(f"correct answer:{correct_answer}")
                print(f"edge entropy rewards:{edge_rewards}")

            utility_loss = torch.mean(torch.stack(loss_list))
        total_loss = utility_loss
        optimizer.zero_grad()
        if not total_loss.requires_grad:
            raise RuntimeError(
                "Graph training loss is not differentiable. A zero-edge sample "
                "must still retain full-graph log-prob or IB gradients."
            )
        total_loss.backward()
        optimizer.step()
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
        print("loss:", total_loss.item()) # 4.6237263679504395
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    return edge_selector if selector_trained else None
        
