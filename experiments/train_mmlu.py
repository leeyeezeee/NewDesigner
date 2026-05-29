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
from GDesigner.utils.uncertainty import (
    SemanticEntailmentJudge,
    edge_entropy_rewards,
    edge_semantic_loss,
    total_reward_with_edges,
)

async def train(graph:Graph,
            dataset,
            num_iters:int=100,
            num_rounds:int=1,
            lr:float=0.1,
            batch_size:int = 4,
            uncertainty_lambda: float = 0.0,
            imp_per_iterations: int = 5,
            pruning_rate: float = 0.25,
            num_entropy_samples: int = 1,
            semantic_judge_llm_name: str = "gpt-4o-mini",
            semantic_judge_api_key: str = "",
            semantic_judge_base_url: str = "",
            semantic_judge_model_path: str = "",
            semantic_judge_max_concurrency: int = None,
            negative_edge_reward_scale: float = 1.0,
            nonpositive_edge_penalty: float = 0.01,
          ) -> None:
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record
    
    loader = infinite_data_loader()
    effective_num_entropy_samples = max(2, int(num_entropy_samples)) if uncertainty_lambda > 0 else max(1, int(num_entropy_samples))
    use_semantic_edges = uncertainty_lambda > 0 and effective_num_entropy_samples > 1
    semantic_judge = None
    if use_semantic_edges:
        semantic_judge = SemanticEntailmentJudge(
            llm_name=semantic_judge_llm_name,
            api_key=semantic_judge_api_key,
            base_url=semantic_judge_base_url,
            model_path=semantic_judge_model_path,
            max_concurrency=semantic_judge_max_concurrency,
        )
    
    optimizer_params = list(graph.gcn.parameters()) + list(graph.mlp.parameters())
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

        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            realized_graph.temporal_logits = graph.temporal_logits
            realized_graphs.append(realized_graph)
            input_dict = dataset.record_to_input(record)
            input_dicts.append(input_dict)
            answer_log_probs.append(asyncio.create_task(
                realized_graph.arun(
                    input_dict,
                    num_rounds,
                    num_entropy_samples=effective_num_entropy_samples,
                    record_execution_history=use_semantic_edges,
                    track_grad=True,
                )
            ))
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)
        
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []
        
        for raw_answer, log_prob, correct_answer, realized_graph, input_dict in zip(raw_answers, log_probs, correct_answers, realized_graphs, input_dicts):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            correctness_reward = accuracy.get()
            edge_rewards = {}
            if correctness_reward > 0 and use_semantic_edges:
                edge_rewards, _ = await edge_entropy_rewards(
                    realized_graph,
                    input_dict["task"],
                    input_dict,
                    semantic_judge,
                    effective_num_entropy_samples,
                    negative_reward_scale=negative_edge_reward_scale,
                    nonpositive_penalty=nonpositive_edge_penalty,
                )
            realized_graph.clear_execution_history()
            edge_losses = edge_semantic_loss(
                realized_graph.edge_log_probs,
                edge_rewards,
                uncertainty_lambda,
                correctness_reward,
            )
            utility = total_reward_with_edges(correctness_reward, edge_rewards, uncertainty_lambda)
            utilities.append(utility)
            single_loss = - log_prob * correctness_reward
            if edge_losses:
                single_loss = single_loss + torch.sum(torch.stack(edge_losses))
            loss_list.append(single_loss)
            print(f"correct answer:{correct_answer}")
            print(f"edge entropy rewards:{edge_rewards}")
    
        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
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
        print("loss:", total_loss.item()) # 4.6237263679504395
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
        
