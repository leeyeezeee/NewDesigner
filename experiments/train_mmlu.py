import torch
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy

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
    SemanticEntailmentJudge,
    edge_entropy_rewards,
)
from GDesigner.utils.ig_scorer import FinalAnswerScorer, make_target_spec
from experiments.refinement_loss import refinement_regularization_loss
from experiments.teacher_forcing_reward import (
    graph_teacher_forcing_score,
    teacher_forcing_edge_loss,
)

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
            anchor_reg_weight: float = 1.0,
            sparsity_reg_weight: float = 1.0,
            use_graph_tf_reward: bool = False,
            graph_sample_count: int = 5,
            graph_softmax_temperature: float = 1.0,
            edge_tanh_temperature: float = 1.0,
          ):
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record
    
    loader = infinite_data_loader()
    effective_num_entropy_samples = (
        max(2, int(num_entropy_samples))
        if (use_edge_selector or use_graph_tf_reward)
        else max(1, int(num_entropy_samples))
    )
    use_semantic_edges = (use_edge_selector or use_graph_tf_reward) and effective_num_entropy_samples > 1
    batch_entropy_samples = effective_num_entropy_samples if use_semantic_edges else 1
    semantic_judge = None
    if use_semantic_edges:
        semantic_judge = SemanticEntailmentJudge(
            llm_name=semantic_judge_llm_name,
            api_key=semantic_judge_api_key,
            base_url=semantic_judge_base_url,
            model_path=semantic_judge_model_path,
            max_concurrency=semantic_judge_max_concurrency,
        )
    edge_selector = None
    selector_buffer = None
    selector_optimizer = None
    selector_trained = False
    if use_edge_selector and use_semantic_edges:
        edge_selector = EdgeSelector(graph.features.size(1))
        selector_buffer = SelectorReplayBuffer(selector_buffer_size)
        selector_optimizer = torch.optim.Adam(edge_selector.parameters(), lr=1e-3)
    tf_scorer = FinalAnswerScorer() if use_graph_tf_reward else None
    
    optimizer_params = list(graph.gcn.parameters()) + list(graph.mlp.parameters()) + graph.refinement_parameters()
    if graph.optimized_temporal:
        optimizer_params.append(graph.temporal_logits)
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    graph.gcn.train()
    graph.mlp.train()
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
            sample_count = max(1, int(graph_sample_count)) if use_graph_tf_reward else 1
            for _ in range(sample_count):
                realized_graph = copy.deepcopy(graph)
                realized_graph.gcn = graph.gcn
                realized_graph.mlp = graph.mlp
                realized_graph.refinement_weight = graph.refinement_weight
                realized_graph.temporal_logits = graph.temporal_logits
                group_indices.append(len(realized_graphs))
                realized_graphs.append(realized_graph)
                input_dicts.append(input_dict)
                answer_log_probs.append(asyncio.create_task(
                    realized_graph.arun(
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
        answers: List[str] = []
        
        if use_graph_tf_reward:
            graph_groups = []
            score_groups = []
            edge_detail_groups = []
            for correct_answer, group_indices, input_dict in zip(
                correct_answers,
                sample_groups,
                record_input_dicts,
            ):
                assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                target_spec = make_target_spec("mmlu", correct_answer)
                graph_group = []
                score_group = []
                edge_detail_group = []
                for sample_pos, graph_idx in enumerate(group_indices):
                    raw_answer = raw_answers[graph_idx]
                    realized_graph = realized_graphs[graph_idx]
                    graph_score = await graph_teacher_forcing_score(
                        tf_scorer,
                        realized_graph,
                        input_dict,
                        raw_answer,
                        target_spec,
                    )
                    edge_rewards, edge_details = await edge_entropy_rewards(
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
                    )
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
                            "graph_tf_score": graph_score.score,
                            "edge_entropy_rewards": edge_rewards,
                        })
                        print(f"correct answer:{correct_answer}")
                        print(f"edge entropy rewards:{edge_rewards}")
                    graph_group.append(realized_graph)
                    score_group.append(graph_score.score)
                    edge_detail_group.append(edge_details)
                    realized_graph.clear_execution_history()
                graph_groups.append(graph_group)
                score_groups.append(score_group)
                edge_detail_groups.append(edge_detail_group)
            reference_loss = torch.mean(torch.stack(list(log_probs)))
            utility_loss, tf_summaries = teacher_forcing_edge_loss(
                graph_groups,
                score_groups,
                edge_detail_groups,
                reference_loss,
                graph_softmax_temperature=graph_softmax_temperature,
                edge_tanh_temperature=edge_tanh_temperature,
            )
            print("teacher forcing graph summaries:", tf_summaries)
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
                if correctness_reward > 0 and use_semantic_edges:
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
                    )
                    if selector_buffer is not None:
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
                single_loss = -log_prob * correctness_reward
                loss_list.append(single_loss)
                print(f"correct answer:{correct_answer}")
                print(f"edge entropy rewards:{edge_rewards}")

            utility_loss = torch.mean(torch.stack(loss_list))
        reg_loss, anchor_loss, sparse_loss = refinement_regularization_loss(
            realized_graphs,
            utility_loss,
            anchor_reg_weight=anchor_reg_weight,
            sparsity_reg_weight=sparsity_reg_weight,
        )
        total_loss = utility_loss + reg_loss
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
            and (i_iter + 1) % imp_per_iterations == 0
        ):
            temporal_masks, pruned_temporal_idx = graph.prune_temporal_edges(pruning_rate)
            print(f"pruned temporal edges: {pruned_temporal_idx.numel()}")
            print("temporal masks:", temporal_masks.view(graph.num_nodes, graph.num_nodes))

        print("answers:",answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities) # [0.0, 0.0, 0.0, 1.0]
        print("utility loss:", utility_loss.item())
        print("anchor loss:", anchor_loss.item())
        print("sparse loss:", sparse_loss.item())
        print("loss:", total_loss.item()) # 4.6237263679504395
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    return edge_selector if selector_trained else None
        
