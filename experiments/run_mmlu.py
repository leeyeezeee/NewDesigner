import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from typing import Union, Literal, List
import argparse
import random

from GDesigner.graph.graph import Graph
from datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from experiments.train_mmlu import train
from experiments.evaluate_mmlu import evaluate
from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.utils.metrics import write_metrics_record



def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered','Star', 'Mesh',
                                 'FakeFullConnected','FakeRandom','FakeChain','FakeStar','FakeMesh','FakeAGRandom','FakeAGFull'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of optimization iterations. Default 10.")
    parser.add_argument('--imp_per_iterations', type=int, default=5,
                        help="Prune temporal edges every few iterations when --optimized_temporal is set.")
    parser.add_argument('--num_rounds',type=int,default=2,
                        help="Number of optimization/inference rounds for one query. Use at least 2 to measure semantic entropy reduction after communication.")
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help="Rate for temporal edge pruning when --optimized_temporal is set.")
    parser.add_argument('--uncertainty_lambda', type=float, default=0.0,
                        help="Enable per-edge semantic entropy analysis when > 0. It is not added to the training loss.")
    parser.add_argument('--num_entropy_samples', type=int, default=5,
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
                        help="Deprecated compatibility option; normalized edge rewards do not add a zero-gain penalty.")
    parser.add_argument('--selector_buffer_size', type=int, default=512,
                        help="Replay buffer capacity for selector edge samples.")
    parser.add_argument('--selector_entropy_tau', type=float, default=0.2,
                        help="Entropy delta threshold for positive selector labels.")
    parser.add_argument('--llm_name', type=str, default="gpt-4o",
                        help="Model name, None runs the default ChatGPT4")
    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain (the same as dataset name), default 'MMLU'")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="the decision method of the final node")
    parser.add_argument('--metrics_file', type=str, default="result/mmlu.jsonl",
                        help="JSONL file to append final accuracy and cost metrics.")
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
    
    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(mode,len(agent_names))
    limit_questions = 153
    
    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    download()
    dataset_train = MMLUDataset('dev')
    dataset_val = MMLUDataset('val')
    
    if args.optimized_spatial or args.optimized_temporal:
        edge_selector = await train(graph=graph,dataset=dataset_train,num_iters=args.num_iterations,num_rounds=args.num_rounds,
                    lr=args.lr,batch_size=args.batch_size, uncertainty_lambda=args.uncertainty_lambda,
                    imp_per_iterations=args.imp_per_iterations, pruning_rate=args.pruning_rate,
                    num_entropy_samples=args.num_entropy_samples,
                    semantic_judge_llm_name=args.semantic_judge_llm_name,
                    semantic_judge_api_key=args.semantic_judge_api_key,
                    semantic_judge_base_url=args.semantic_judge_base_url,
                    semantic_judge_model_path=args.semantic_judge_model_path,
                    semantic_judge_max_concurrency=args.semantic_judge_max_concurrency,
                    negative_edge_reward_scale=args.negative_edge_reward_scale,
                    nonpositive_edge_penalty=args.nonpositive_edge_penalty,
                    selector_buffer_size=args.selector_buffer_size,
                    selector_entropy_tau=args.selector_entropy_tau)
    else:
        edge_selector = None

    eval_metrics = await evaluate(graph=graph,dataset=dataset_val,num_rounds=args.num_rounds,limit_questions=limit_questions,eval_batch_size=args.batch_size,edge_selector=edge_selector)
    score = eval_metrics["accuracy"]
    print(f"Final Eval Accuracy: {score}")
    print(f"Final Avg Edges: {eval_metrics['avg_edges']}")
    write_metrics_record(args.metrics_file, {
        "dataset": "mmlu",
        "accuracy": score,
        "total_solved": eval_metrics["total_solved"],
        "total_executed": eval_metrics["total_executed"],
        "avg_edges": eval_metrics["avg_edges"],
        "llm_name": args.llm_name,
    })



def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star'],Literal['Mesh'],
                          Literal['FakeFullConnected'],Literal['FakeRandom'],Literal['FakeChain'],Literal['FakeStar'],Literal['FakeMesh'],Literal['FakeAGRandom'],Literal['FakeAGFull']],
               N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0]*N for _ in range(N)]
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
    
    def generate_mesh_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(0, N):
            for j in range(i+1,N):
                adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1,N):
            adj_matrix[0][i] = 1
        return adj_matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Normal'}]
    elif mode=='FullConnected' or mode == 'FakeFullConnected' or mode=='FakeAGFull':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random' or mode == 'FakeRandom' or mode == 'FakeAGRandom':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain' or mode == 'FakeChain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Mesh' or mode=='FakeMesh':
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star' or mode=='FakeStar':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':None} for i in range(N)]
        
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == "__main__":
    asyncio.run(main())
