import argparse
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from GDesigner.tools.coding.python_executor import PyExecutor
from GDesigner.utils.const import GDesigner_ROOT
from experiments.unsup_logprob_runner import (
    add_common_unsup_args,
    finalize_unsup_args,
    humaneval_answer_parser,
    load_jsonl,
    run_unsup_stage,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner HumanEval unsupervised logprob stage")
    parser.add_argument("--dataset_json", type=str, default="datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star"],
    )
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--domain", type=str, default="humaneval")
    parser.add_argument("--agent_names", nargs="+", type=str, default=["CodeWriting"])
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[5])
    parser.add_argument("--decision_method", type=str, default="FinalWriteCode")
    parser.add_argument("--refine_rank", type=int, default=4)
    parser.add_argument("--edge_bias_scale", type=float, default=0.5)
    parser.add_argument("--humaneval_timeout", type=int, default=100)
    add_common_unsup_args(
        parser,
        dataset_name="humaneval",
        unsup_data="ExpanData/unlabeled/humaneval-py.jsonl",
        stage1_checkpoint="result/checkpoints/humaneval.pt",
        checkpoint_file="result/checkpoints/humaneval_unsup.pt",
        metrics_file="result/humaneval_unsup.jsonl",
    )
    args = parser.parse_args()
    os.makedirs(GDesigner_ROOT / "result", exist_ok=True)
    return finalize_unsup_args(parser, args)


async def main():
    args = parse_args()
    eval_records = load_jsonl(args.dataset_json)
    executor = PyExecutor()

    def humaneval_correct(predicted: str, target: str) -> bool:
        is_solved, _feedback, _state = executor.execute(
            predicted,
            [target],
            timeout=args.humaneval_timeout,
        )
        return bool(is_solved)

    await run_unsup_stage(
        args,
        dataset_name="humaneval",
        graph_domain=args.domain,
        eval_records=eval_records,
        record_to_input=lambda record: {"task": record["prompt"]},
        record_to_target=lambda record: record["test"],
        answer_parser=humaneval_answer_parser,
        correctness_fn=humaneval_correct,
    )


if __name__ == "__main__":
    asyncio.run(main())
