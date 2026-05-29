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
from GDesigner.utils.metrics import write_metrics_record
from GDesigner.utils.uncertainty import (
    SemanticEntailmentJudge,
    edge_entropy_rewards,
    edge_semantic_loss,
    total_reward_with_edges,
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
    parser.add_argument('--uncertainty_lambda', type=float, default=0.0,
                        help="Weight for edge-level semantic entropy reward. Default 0 keeps original utility.")
    parser.add_argument('--num_entropy_samples', type=int, default=1,
                        help="Samples per agent before and after communication for semantic entropy. Automatically raised to 2 when uncertainty_lambda > 0.")
    parser.add_argument('--semantic_judge_llm_name', type=str, default="gpt-4o-mini",
                        help="OpenAI-compatible semantic judge model name. Independent from --llm_name.")
    parser.add_argument('--semantic_judge_api_key', type=str, default="",
                        help="Semantic judge API key. For local vLLM, EMPTY is usually enough.")
    parser.add_argument('--semantic_judge_base_url', type=str, default="",
                        help="Semantic judge OpenAI-compatible base URL. Use http://localhost:8000/v1 for local vLLM.")
    parser.add_argument('--semantic_judge_model_path', type=str, default="",
                        help="Optional judge model name override kept for backward compatibility.")
    parser.add_argument('--semantic_judge_max_concurrency', type=int, default=None,
                        help="Maximum concurrent semantic judge API requests. Defaults to SEMANTIC_JUDGE_MAX_CONCURRENCY or 16.")
    parser.add_argument('--negative_edge_reward_scale', type=float, default=1.0,
                        help="Scale for negative edge rewards when an edge increases semantic entropy.")
    parser.add_argument('--nonpositive_edge_penalty', type=float, default=0.01,
                        help="Extra penalty when a selected edge does not reduce semantic entropy.")
    parser.add_argument('--num_iterations', type=int, default = 10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="humaneval",help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalWriteCode',
                        help='The decison method of the GDesigner')
    parser.add_argument('--metrics_file', type=str, default="result/humaneval.jsonl",
                        help="JSONL file to overwrite with final accuracy and cost metrics.")
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
    optimizer_params = list(graph.gcn.parameters()) + list(graph.mlp.parameters())
    if graph.optimized_temporal:
        optimizer_params.append(graph.temporal_logits)
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr)
    effective_num_entropy_samples = max(2, int(args.num_entropy_samples)) if args.uncertainty_lambda > 0 else max(1, int(args.num_entropy_samples))
    optimize_enabled = args.optimized_spatial or args.optimized_temporal
    use_semantic_edges_for_training = optimize_enabled and args.uncertainty_lambda > 0 and effective_num_entropy_samples > 1
    semantic_judge = None
    if use_semantic_edges_for_training:
        semantic_judge = SemanticEntailmentJudge(
            llm_name=args.semantic_judge_llm_name,
            api_key=args.semantic_judge_api_key,
            base_url=args.semantic_judge_base_url,
            model_path=args.semantic_judge_model_path,
            max_concurrency=args.semantic_judge_max_concurrency,
        )
    
    num_batches = int(len(dataset)/args.batch_size)
    total_solved, total_executed = (0, 0)
    accuracy = 0.0
    for i_batch in range(num_batches):
        train_updates_enabled = optimize_enabled and i_batch < args.num_iterations
        use_semantic_edges = use_semantic_edges_for_training and train_updates_enabled
        batch_entropy_samples = effective_num_entropy_samples if use_semantic_edges else 1
        print(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        answer_log_probs = []
        tests = []
        realized_graphs = []
        input_dicts = []
        
        current_batch = dataloader(dataset,args.batch_size,i_batch)
        if current_batch is None:
            print("No more data available.")
            break
        
        for i_record, record in enumerate(current_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            realized_graph.temporal_logits = graph.temporal_logits
            realized_graphs.append(realized_graph)
            task = record["prompt"]
            test = record["test"]
            tests.append(test)
            input_dict = {"task": task}
            input_dicts.append(input_dict)
            answer_log_probs.append(asyncio.create_task(
                realized_graph.arun(
                    input_dict,
                    args.num_rounds,
                    num_entropy_samples=batch_entropy_samples,
                    record_execution_history=use_semantic_edges,
                    track_grad=train_updates_enabled,
                )
            ))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []

        for task, answer, log_prob, test, realized_graph, input_dict in zip(current_batch, raw_answers, log_probs, tests, realized_graphs, input_dicts):
            if not isinstance(answer,list):
                raise TypeError(f"Expected a list for the answer, but got {type(answer).__name__}")
            answer = answer[0].lstrip("```python\n").rstrip("\n```")
            is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            accuracy = total_solved/ total_executed

            edge_rewards = {}
            if is_solved and use_semantic_edges:
                edge_rewards, _ = await edge_entropy_rewards(
                    realized_graph,
                    task["prompt"],
                    input_dict,
                    semantic_judge,
                    effective_num_entropy_samples,
                    negative_reward_scale=args.negative_edge_reward_scale,
                    nonpositive_penalty=args.nonpositive_edge_penalty,
                )
            realized_graph.clear_execution_history()
            edge_losses = edge_semantic_loss(
                realized_graph.edge_log_probs,
                edge_rewards,
                args.uncertainty_lambda,
                is_solved,
            )
            utility = total_reward_with_edges(is_solved, edge_rewards, args.uncertainty_lambda)
            utilities.append(utility)
            single_loss = -log_prob * is_solved
            if edge_losses:
                single_loss = single_loss + torch.sum(torch.stack(edge_losses))
            loss_list.append(single_loss)

        total_loss = torch.mean(torch.stack(loss_list))
        if train_updates_enabled:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
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

        if i_batch+1 == args.num_iterations:
            total_solved = 0
            total_executed = 0
            accuracy = 0.0
            graph.gcn.eval()
            graph.mlp.eval()
            print("Start Eval")
            
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    print(f"Final Eval Accuracy: {accuracy}")
    print(f"Final Cost {Cost.instance().value}")
    print(f"Final PromptTokens {PromptTokens.instance().value}")
    print(f"Final CompletionTokens {CompletionTokens.instance().value}")
    write_metrics_record(args.metrics_file, {
        "dataset": "humaneval",
        "accuracy": accuracy,
        "total_solved": total_solved,
        "total_executed": total_executed,
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
    asyncio.run(main())
