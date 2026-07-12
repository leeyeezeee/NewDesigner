import argparse
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from datasets.gsm8k_dataset import gsm_get_predict, svamp_data_process
from GDesigner.tools.reader.readers import JSONReader
from GDesigner.utils.const import GDesigner_ROOT
from experiments.teacher_forcing_reward import add_teacher_forcing_reward_args
from math_dataset_runner import numeric_correct, run_math_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on SVAMP")
    parser.add_argument("--dataset_json", type=str, default="datasets/SVAMP/SVAMP.json")
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
    parser.add_argument("--domain", type=str, default="gsm8k")
    parser.add_argument("--agent_names", nargs="+", type=str, default=["MathSolver"])
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[4])
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--metrics_file", type=str, default="result/svamp.jsonl")
    parser.add_argument("--checkpoint_file", type=str, default="result/checkpoints/svamp.pt",
                        help="Path to overwrite with the trained graph checkpoint.")
    parser.add_argument("--use_edge_selector", action="store_true",
                        help="Enable final-agent teacher-logprob/execution IG selector training and selector pruning during evaluation.")
    parser.add_argument("--num_entropy_samples", type=int, default=1,
                        help="Deprecated for final-agent teacher-logprob IG; non-HumanEval IG scores final-agent teacher answers directly.")
    # KLE temporarily disabled; keep this hyperparameter ready for future re-enable.
    # parser.add_argument("--kle_heat_t", type=float, default=0.3,
    #                     help="Heat-kernel lengthscale for KHEAT uncertainty.")
    parser.add_argument("--semantic_judge_llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--semantic_judge_api_key", type=str, default="")
    parser.add_argument("--semantic_judge_base_url", type=str, default="")
    parser.add_argument("--semantic_judge_model_path", type=str, default="")
    parser.add_argument("--semantic_judge_max_concurrency", type=int, default=None)
    parser.add_argument("--negative_edge_reward_scale", type=float, default=1.0)
    parser.add_argument("--nonpositive_edge_penalty", type=float, default=0.01)
    parser.add_argument("--selector_buffer_size", type=int, default=512)
    parser.add_argument("--selector_ig_tau", type=float, default=0.0)
    parser.add_argument("--refine_rank", type=int, default=4,
                        help="Rank used by the refined adjacency decoder.")
    parser.add_argument("--anchor_reg_weight", type=float, default=1.0,
                        help="Weight for G-Designer refined adjacency anchor regularization.")
    parser.add_argument("--sparsity_reg_weight", type=float, default=1.0,
                        help="Weight for G-Designer refined adjacency nuclear-norm sparsity regularization.")
    add_teacher_forcing_reward_args(parser)
    parser.add_argument("--optimized_spatial", action="store_true")
    parser.add_argument("--optimized_temporal", action="store_true")
    args = parser.parse_args()
    os.makedirs(GDesigner_ROOT / "result", exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main():
    args = parse_args()
    dataset = svamp_data_process(JSONReader.parse_file(args.dataset_json))
    await run_math_dataset(
        args,
        dataset,
        dataset_name="svamp",
        graph_domain=args.domain,
        answer_parser=gsm_get_predict,
        correctness_fn=numeric_correct,
    )


if __name__ == "__main__":
    asyncio.run(main())
