import argparse
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.utils.const import GDesigner_ROOT
from math_dataset_runner import choice_correct, run_math_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on AQuA")
    parser.add_argument("--dataset_json", type=str, default="datasets/AQuA/AQuA.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star"],
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--imp_per_iterations", type=int, default=5,
                        help="Prune temporal edges every few iterations when --optimized_temporal is set.")
    parser.add_argument("--pruning_rate", type=float, default=0.25,
                        help="Rate for temporal edge pruning when --optimized_temporal is set.")
    parser.add_argument("--domain", type=str, default="aqua")
    parser.add_argument("--agent_names", nargs="+", type=str, default=["MathSolver_aqua"])
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[4])
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--metrics_file", type=str, default="result/aqua.jsonl")
    parser.add_argument("--uncertainty_lambda", type=float, default=0.0)
    parser.add_argument("--num_entropy_samples", type=int, default=1)
    parser.add_argument("--semantic_judge_llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--semantic_judge_api_key", type=str, default="")
    parser.add_argument("--semantic_judge_base_url", type=str, default="")
    parser.add_argument("--semantic_judge_model_path", type=str, default="")
    parser.add_argument("--semantic_judge_max_concurrency", type=int, default=None)
    parser.add_argument("--negative_edge_reward_scale", type=float, default=1.0)
    parser.add_argument("--nonpositive_edge_penalty", type=float, default=0.01)
    parser.add_argument("--optimized_spatial", action="store_true")
    parser.add_argument("--optimized_temporal", action="store_true")
    args = parser.parse_args()
    os.makedirs(GDesigner_ROOT / "result", exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main():
    args = parse_args()
    dataset = aqua_data_process(JSONLReader.parse_file(args.dataset_json))
    await run_math_dataset(
        args,
        dataset,
        dataset_name="aqua",
        graph_domain=args.domain,
        answer_parser=aqua_get_predict,
        correctness_fn=choice_correct,
    )


if __name__ == "__main__":
    asyncio.run(main())
