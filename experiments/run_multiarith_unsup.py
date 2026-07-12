import argparse
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from GDesigner.utils.const import GDesigner_ROOT
from datasets.gsm8k_dataset import gsm_get_predict, multiarith_data_process
from experiments.unsup_logprob_runner import (
    add_common_unsup_args,
    finalize_unsup_args,
    load_json,
    numeric_correct,
    run_unsup_stage,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner MultiArith unsupervised logprob stage")
    parser.add_argument("--dataset_json", type=str, default="datasets/MultiArith/MultiArith.json")
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star"],
    )
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--domain", type=str, default="gsm8k")
    parser.add_argument("--agent_names", nargs="+", type=str, default=["MathSolver"])
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[5])
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--refine_rank", type=int, default=4)
    add_common_unsup_args(
        parser,
        dataset_name="multiarith",
        unsup_data="ExpanData/unlabeled/MultiArith.jsonl",
        stage1_checkpoint="result/checkpoints/multiarith.pt",
        checkpoint_file="result/checkpoints/multiarith_unsup.pt",
        metrics_file="result/multiarith_unsup.jsonl",
    )
    args = parser.parse_args()
    os.makedirs(GDesigner_ROOT / "result", exist_ok=True)
    return finalize_unsup_args(parser, args)


async def main():
    args = parse_args()
    eval_records = multiarith_data_process(load_json(args.dataset_json))
    await run_unsup_stage(
        args,
        dataset_name="multiarith",
        graph_domain=args.domain,
        eval_records=eval_records,
        record_to_input=lambda record: {"task": record["task"]},
        record_to_target=lambda record: record["answer"],
        answer_parser=gsm_get_predict,
        correctness_fn=numeric_correct,
    )


if __name__ == "__main__":
    asyncio.run(main())
