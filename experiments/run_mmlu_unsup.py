import argparse
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from GDesigner.utils.const import GDesigner_ROOT
from datasets.MMLU.download import download
from datasets.mmlu_dataset import MMLUDataset
from experiments.unsup_logprob_runner import (
    add_common_unsup_args,
    exact_correct,
    finalize_unsup_args,
    run_unsup_stage,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner MMLU unsupervised logprob stage")
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"],
    )
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--domain", type=str, default="mmlu")
    parser.add_argument("--agent_names", nargs="+", type=str, default=["AnalyzeAgent"])
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[6])
    parser.add_argument("--decision_method", type=str, default="FinalRefer")
    parser.add_argument("--refine_rank", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    add_common_unsup_args(
        parser,
        dataset_name="mmlu",
        unsup_data="ExpanData/unlabeled/mmlu.jsonl",
        stage1_checkpoint="result/checkpoints/mmlu.pt",
        checkpoint_file="result/checkpoints/mmlu_unsup.pt",
        metrics_file="result/mmlu_unsup.jsonl",
    )
    args = parser.parse_args()
    os.makedirs(GDesigner_ROOT / "result", exist_ok=True)
    return finalize_unsup_args(parser, args)


async def main():
    args = parse_args()
    if args.download:
        download()
    eval_dataset = MMLUDataset("val")
    await run_unsup_stage(
        args,
        dataset_name="mmlu",
        graph_domain=args.domain,
        eval_records=[eval_dataset[i] for i in range(len(eval_dataset))],
        record_to_input=eval_dataset.record_to_input,
        record_to_target=eval_dataset.record_to_target_answer,
        answer_parser=eval_dataset.postprocess_answer,
        correctness_fn=exact_correct,
    )


if __name__ == "__main__":
    from experiments.crash_logging import run_async_with_crash_logging
    run_async_with_crash_logging(main)
