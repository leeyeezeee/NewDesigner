import sys
import os
import argparse
import yaml
import time
import asyncio
import torch
import copy
from typing import List,Union,Literal
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from GDesigner.graph.graph import Graph
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.tools.coding.python_executor import PyExecutor
from GDesigner.utils.globals import Time
from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
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
    add_teacher_forcing_reward_args,
    edge_information_gain_loss,
    graph_correctness_advantage_edge_loss,
)
from experiments.edge_training_log import (
    append_edge_training_details,
    reset_edge_training_log,
    resolve_edge_training_log_file,
    resolve_question_id,
)

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on HumanEval")
    parser.add_argument("--dataset_json", type=str, default="datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4-1106-preview")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--num_rounds',type=int,default=2,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help="Rate for temporal edge pruning when --optimized_temporal is set.")
    parser.add_argument('--imp_per_iterations', type=int, default=5,
                        help="Prune temporal edges every few iterations when --optimized_temporal is set.")
    parser.add_argument('--use_edge_selector', action='store_true',
                        help="Enable final-agent teacher-logprob/execution IG selector training and selector pruning during evaluation.")
    parser.add_argument('--num_entropy_samples', type=int, default=1,
                        help="Deprecated for final-agent teacher-logprob IG; non-HumanEval IG scores final-agent teacher answers directly.")
    # KLE temporarily disabled; keep this hyperparameter ready for future re-enable.
    # parser.add_argument('--kle_heat_t', type=float, default=0.3,
    #                     help="Heat-kernel lengthscale for KHEAT uncertainty.")
    parser.add_argument('--semantic_judge_llm_name', type=str, default="gpt-4o-mini",
                        help="OpenAI-compatible semantic judge model name. Independent from --llm_name.")
    parser.add_argument('--semantic_judge_api_key', type=str, default="",
                        help="Semantic judge API key. For local vLLM, EMPTY is usually enough.")
    parser.add_argument('--semantic_judge_base_url', type=str, default="",
                        help="Semantic judge OpenAI-compatible base URL. Use http://localhost:8000/v1 for local vLLM.")
    parser.add_argument('--semantic_judge_model_path', type=str, default="",
                        help="Optional judge model name override kept for backward compatibility.")
    parser.add_argument('--semantic_judge_max_concurrency', type=int, default=None,
                        help="Maximum concurrent semantic judge API requests. Defaults to SEMANTIC_JUDGE_MAX_CONCURRENCY or 64.")
    parser.add_argument('--negative_edge_reward_scale', type=float, default=1.0,
                        help="Scale for negative edge rewards when an edge has negative IG gain.")
    parser.add_argument('--nonpositive_edge_penalty', type=float, default=0.01,
                        help="Deprecated compatibility option; normalized edge rewards do not add a zero-gain penalty.")
    parser.add_argument('--selector_buffer_size', type=int, default=512,
                        help="Replay buffer capacity for selector edge samples.")
    parser.add_argument('--selector_ig_tau', type=float, default=0.0,
                        help="IG gain threshold for positive selector labels.")
    add_teacher_forcing_reward_args(parser)
    parser.add_argument('--num_iterations', type=int, default = 10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="humaneval",help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalWriteCode',
                        help='The decison method of the GDesigner')
    parser.add_argument('--metrics_file', type=str, default="result/humaneval.jsonl",
                        help="JSONL file to append final accuracy and cost metrics.")
    parser.add_argument('--checkpoint_file', type=str, default="result/checkpoints/humaneval.pt",
                        help="Path to overwrite with the trained graph checkpoint.")
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    
    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args

async def main():
    args = parse_args()
    apply_agent_backend_args(args)
    dataset = JSONLReader.parse_file(args.dataset_json)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode,len(agent_names))
    graph = Graph(domain="humaneval",
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    graph.gcn.train()
    graph.mlp.train()
    edge_training_log_file = resolve_edge_training_log_file("humaneval")
    reset_edge_training_log(edge_training_log_file)
    optimizer_params = (
        list(graph.gcn.parameters())
        + list(graph.mlp.parameters())
        + graph.spatial_parameters()
    )
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
            or bool(edge_training_log_file)
        )
    )
    semantic_judge = None
    edge_selector = None
    selector_buffer = None
    selector_optimizer = None
    selector_trained = False
    if args.use_edge_selector and use_semantic_edges_for_analysis:
        edge_selector = EdgeSelector(graph.features.size(1))
        selector_buffer = SelectorReplayBuffer(args.selector_buffer_size)
        selector_optimizer = torch.optim.Adam(edge_selector.parameters(), lr=1e-3)
    tf_scorer = (
        FinalAnswerScorer()
        if (
            args.use_edge_selector
            or edge_ig_reward_lambda != 0.0
            or bool(edge_training_log_file)
        )
        else None
    )
    graph_semaphore = make_graph_semaphore(args.max_concurrent_graphs)
    
    num_batches = int(len(dataset)/args.batch_size)
    total_solved, total_executed = (0, 0)
    total_edges, edge_samples = (0, 0)
    accuracy = 0.0
    for i_batch in range(num_batches):
        train_updates_enabled = optimize_enabled and i_batch < args.num_iterations
        edge_ig_measurement_enabled = (
            train_updates_enabled
            and i_batch >= max(0, int(args.edge_ig_warmup_iterations))
        )
        iteration_edge_ig_reward_lambda = (
            edge_ig_reward_lambda if edge_ig_measurement_enabled else 0.0
        )
        use_semantic_edges = use_semantic_edges_for_analysis and train_updates_enabled
        batch_entropy_samples = 1
        batch_edge_selector = edge_selector if (selector_trained and not train_updates_enabled) else None
        print(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        answer_log_probs = []
        tests = []
        realized_graphs = []
        input_dicts = []
        record_input_dicts = []
        question_ids = []
        sample_groups = []

        current_batch = dataloader(dataset,args.batch_size,i_batch)
        if current_batch is None:
            print("No more data available.")
            break

        for i_record, record in enumerate(current_batch):
            question_ids.append(resolve_question_id(
                record, i_batch * args.batch_size + i_record
            ))
            task = record["prompt"]
            test = record["test"]
            tests.append(test)
            input_dict = {"task": task}
            record_input_dicts.append(input_dict)
            group_indices = []
            sample_count = (
                max(1, int(args.graph_sample_count))
                if (use_multi_graph_reward and train_updates_enabled)
                else 1
            )
            for _ in range(sample_count):
                realized_graph = copy.deepcopy(graph)
                realized_graph.gcn = graph.gcn
                realized_graph.node_self_projection = graph.node_self_projection
                realized_graph.node_feature_norm = graph.node_feature_norm
                realized_graph.mlp = graph.mlp
                realized_graph.spatial_affinity_weight = graph.spatial_affinity_weight
                realized_graph.temporal_logits = graph.temporal_logits
                group_indices.append(len(realized_graphs))
                realized_graphs.append(realized_graph)
                input_dicts.append(input_dict)
                answer_log_probs.append(asyncio.create_task(
                    limited_graph_arun(
                        graph_semaphore,
                        realized_graph,
                        input_dict,
                        args.num_rounds,
                        num_entropy_samples=batch_entropy_samples,
                        record_execution_history=use_semantic_edges,
                        track_grad=train_updates_enabled,
                        edge_selector=batch_edge_selector,
                        track_graph_tokens=(
                            use_multi_graph_reward
                            and train_updates_enabled
                            and args.graph_token_cost_lambda != 0.0
                        ),
                    )
                ))
            sample_groups.append(group_indices)
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[dict] = []

        if use_multi_graph_reward and train_updates_enabled:
            graph_groups = []
            graph_log_prob_groups = []
            correctness_groups = []
            graph_token_groups = []
            edge_detail_groups = []
            graph_tf_corrects: List[float] = []
            graph_tf_edge_counts: List[float] = []
            for record, test, group_indices, input_dict, question_id in zip(
                current_batch,
                tests,
                sample_groups,
                record_input_dicts,
                question_ids,
            ):
                target_spec = make_target_spec("humaneval", tests=[test])
                graph_group = []
                graph_log_prob_group = []
                correctness_group = []
                graph_token_group = []
                edge_detail_group = []
                for sample_pos, graph_idx in enumerate(group_indices):
                    realized_graph = realized_graphs[graph_idx]
                    raw_answer = raw_answers[graph_idx]
                    if not isinstance(raw_answer,list):
                        raise TypeError(f"Expected a list for the answer, but got {type(raw_answer).__name__}")
                    answer = raw_answer[0].lstrip("```python\n").rstrip("\n```")
                    is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
                    graph_tf_corrects.append(float(is_solved))
                    graph_tf_edge_counts.append(realized_graph.mean_spatial_edges_per_round)
                    needs_edge_details = (
                        edge_ig_measurement_enabled
                        and bool(realized_graph.edge_log_probs)
                        and (
                            iteration_edge_ig_reward_lambda != 0.0
                            or selector_buffer is not None
                            or bool(edge_training_log_file)
                        )
                    )
                    if needs_edge_details:
                        _edge_rewards, edge_details = await edge_entropy_rewards(
                            realized_graph,
                            record["prompt"],
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
                    append_edge_training_details(
                        edge_training_log_file,
                        question_id=question_id,
                        edge_details=edge_details,
                    )
                    if selector_buffer is not None:
                        selector_buffer.add_many(build_edge_selector_examples(
                            realized_graph,
                            record["prompt"],
                            edge_details,
                            args.selector_ig_tau,
                        ))
                    if sample_pos == 0:
                        total_solved = total_solved + is_solved
                        total_executed = total_executed + 1
                        accuracy = total_solved / total_executed
                        utilities.append({
                            "correctness": is_solved,
                        })
                    graph_group.append(realized_graph)
                    graph_log_prob_group.append(log_probs[graph_idx])
                    correctness_group.append(float(is_solved))
                    graph_token_group.append(float(
                        getattr(realized_graph, "graph_token_usage", {}).get(
                            "total_tokens", 0
                        )
                    ))
                    edge_detail_group.append(edge_details)
                    realized_graph.clear_execution_history()
                graph_groups.append(graph_group)
                graph_log_prob_groups.append(graph_log_prob_group)
                correctness_groups.append(correctness_group)
                graph_token_groups.append(graph_token_group)
                edge_detail_groups.append(edge_detail_group)
            reference_loss = torch.mean(torch.stack(list(log_probs)))
            utility_loss, tf_summaries = graph_correctness_advantage_edge_loss(
                graph_groups,
                graph_log_prob_groups,
                correctness_groups,
                edge_detail_groups,
                reference_loss,
                graph_token_groups=graph_token_groups,
                graph_token_cost_lambda=args.graph_token_cost_lambda,
                edge_tanh_temperature=args.edge_tanh_temperature,
                edge_ig_reward_lambda=iteration_edge_ig_reward_lambda,
                edge_ig_discount_factor=args.edge_ig_discount_factor,
                advantage_epsilon=args.graph_advantage_epsilon,
            )
            if graph_tf_corrects:
                avg_correct = sum(graph_tf_corrects) / len(graph_tf_corrects)
                avg_edges = sum(graph_tf_edge_counts) / len(graph_tf_edge_counts)
                avg_adv_variance = (
                    sum(summary["graph_reward_variance"] for summary in tf_summaries)
                    / len(tf_summaries)
                    if tf_summaries
                    else 0.0
                )
                avg_adv_std = (
                    sum(summary["graph_reward_std"] for summary in tf_summaries)
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
            for graph_idx, (task, answer, log_prob, test, realized_graph, input_dict, question_id) in enumerate(zip(current_batch, raw_answers, log_probs, tests, realized_graphs, input_dicts, question_ids)):
                if not isinstance(answer,list):
                    raise TypeError(f"Expected a list for the answer, but got {type(answer).__name__}")
                answer = answer[0].lstrip("```python\n").rstrip("\n```")
                is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                accuracy = total_solved/ total_executed
                if not train_updates_enabled:
                    total_edges += realized_graph.mean_spatial_edges_per_round
                    edge_samples += 1

                edge_rewards = {}
                edge_details = {}
                if (
                    use_semantic_edges
                    and edge_ig_measurement_enabled
                    and (
                        is_solved
                        or iteration_edge_ig_reward_lambda != 0.0
                        or bool(edge_training_log_file)
                    )
                    and bool(realized_graph.edge_log_probs)
                ):
                    edge_rewards, edge_details = await edge_entropy_rewards(
                        realized_graph,
                        task["prompt"],
                        input_dict,
                        semantic_judge,
                        effective_num_entropy_samples,
                        negative_reward_scale=args.negative_edge_reward_scale,
                        nonpositive_penalty=args.nonpositive_edge_penalty,
                        kle_heat_t=getattr(args, "kle_heat_t", 0.3),
                        target_spec=make_target_spec("humaneval", tests=[test]),
                        ig_scorer=tf_scorer,
                    )
                    if selector_buffer is not None and is_solved:
                        selector_buffer.add_many(build_edge_selector_examples(
                            realized_graph,
                            task["prompt"],
                            edge_details,
                            args.selector_ig_tau,
                        ))
                append_edge_training_details(
                    edge_training_log_file,
                    question_id=question_id,
                    edge_details=edge_details,
                )
                realized_graph.clear_execution_history()
                utility = {
                    "correctness": is_solved,
                    "edge_entropy_rewards": edge_rewards,
                }
                utilities.append(utility)
                single_loss = -log_prob * float(is_solved)
                if iteration_edge_ig_reward_lambda != 0.0:
                    edge_ig_loss, edge_ig_summary = edge_information_gain_loss(
                        realized_graph,
                        edge_details,
                        log_prob,
                        edge_tanh_temperature=args.edge_tanh_temperature,
                        edge_ig_reward_lambda=iteration_edge_ig_reward_lambda,
                        edge_ig_discount_factor=args.edge_ig_discount_factor,
                    )
                    single_loss = single_loss + edge_ig_loss
                    utility["edge_ig_loss_summary"] = edge_ig_summary
                loss_list.append(single_loss)

            utility_loss = torch.mean(torch.stack(loss_list))
        total_loss = utility_loss
        if train_updates_enabled:
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

        if i_batch+1 == args.num_iterations:
            save_graph_checkpoint(
                graph,
                args.checkpoint_file,
                dataset="humaneval",
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
        "dataset": "humaneval",
        "accuracy": accuracy,
        "total_solved": total_solved,
        "total_executed": total_executed,
        "avg_edges": avg_edges,
        "llm_name": args.llm_name,
    })



def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']],
               N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix

    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Programming Expert'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == '__main__':
    from experiments.crash_logging import run_async_with_crash_logging
    run_async_with_crash_logging(main)
