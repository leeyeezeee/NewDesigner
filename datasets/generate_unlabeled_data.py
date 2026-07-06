from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "default_input": "datasets/gsm8k/gsm8k.jsonl",
        "output_name": "gsm8k.jsonl",
        "format": "jsonl",
        "required": ["question"],
        "forbidden": ["answer", "step", "rationale", "correct", "correct_answer"],
        "description": (
            "Grade-school arithmetic word problems. Each entry has only a question. "
            "The question should ask for a single numeric answer, but do not include the answer."
        ),
    },
    "svamp": {
        "default_input": "datasets/SVAMP/SVAMP.json",
        "output_name": "SVAMP.jsonl",
        "format": "jsonl",
        "required": ["Body", "Question"],
        "forbidden": ["Answer", "answer", "step", "rationale", "correct"],
        "description": (
            "SVAMP-style arithmetic word problems split into Body and Question. "
            "Do not include the numeric answer."
        ),
    },
    "multiarith": {
        "default_input": "datasets/MultiArith/MultiArith.json",
        "output_name": "MultiArith.jsonl",
        "format": "jsonl",
        "required": ["sQuestion"],
        "forbidden": ["lSolutions", "answer", "step", "rationale", "correct"],
        "description": (
            "MultiArith-style multi-step arithmetic word problems. Each entry has only sQuestion."
        ),
    },
    "aqua": {
        "default_input": "datasets/AQuA/AQuA.jsonl",
        "output_name": "AQuA.jsonl",
        "format": "jsonl",
        "required": ["question", "options"],
        "forbidden": ["rationale", "correct", "answer", "step"],
        "description": (
            "AQuA-style multiple-choice quantitative reasoning. Each entry has question and "
            "five options labeled A) through E). Do not include rationale or correct."
        ),
    },
    "mmlu": {
        "default_input": "datasets/MMLU/data/dev",
        "output_name": "mmlu.jsonl",
        "format": "jsonl",
        "required": ["question", "A", "B", "C", "D"],
        "forbidden": ["correct_answer", "correct", "answer", "rationale"],
        "description": (
            "MMLU-style four-choice knowledge/reasoning questions. Each entry has question, "
            "A, B, C, and D. Do not include the correct option."
        ),
    },
    "humaneval": {
        "default_input": "datasets/humaneval/humaneval-py.jsonl",
        "output_name": "humaneval-py.jsonl",
        "format": "jsonl",
        "required": ["name", "language", "prompt", "entry_point", "stop_tokens"],
        "forbidden": ["test", "canonical_solution", "solution", "answer"],
        "description": (
            "HumanEval-style Python programming tasks. Each entry has a function prompt with "
            "signature/docstring, entry_point, language='py', and stop_tokens. Do not include tests "
            "or solutions."
        ),
    },
}

DEFAULT_STOP_TOKENS = ["\ndef", "\n#", "\nif", "\nclass"]


def _client_kwargs(base_url: str, api_key: str) -> Dict[str, Any]:
    kwargs = {"api_key": api_key or "EMPTY", "timeout": 1200.0}
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else [data]


def _load_mmlu(path: Path) -> List[Dict[str, Any]]:
    paths: List[Path]
    if path.is_dir():
        paths = sorted(path.glob("*.csv"))
    else:
        paths = [path]
    rows: List[Dict[str, Any]] = []
    for csv_path in paths:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 5:
                    continue
                rows.append({
                    "question": row[0],
                    "A": row[1],
                    "B": row[2],
                    "C": row[3],
                    "D": row[4],
                })
    return rows


def load_seed_records(dataset: str, input_path: Path, max_records: int) -> List[Dict[str, Any]]:
    if not input_path.exists():
        return []
    if dataset == "mmlu":
        records = _load_mmlu(input_path)
    elif input_path.suffix.lower() == ".jsonl":
        records = _load_jsonl(input_path)
    elif input_path.suffix.lower() == ".json":
        records = _load_json(input_path)
    else:
        raise ValueError(f"Unsupported input format for {input_path}")
    return [strip_to_schema(dataset, record) for record in records[:max_records]]


def training_seed_limit(dataset: str, args) -> int:
    if args.seed_train_records is not None:
        return max(1, int(args.seed_train_records))
    if dataset == "mmlu":
        return max(1, int(args.seed_examples))
    return max(1, int(args.train_batch_size) * int(args.train_num_iterations))


def strip_to_schema(dataset: str, record: Dict[str, Any]) -> Dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    clean = {key: record[key] for key in spec["required"] if key in record}
    if dataset == "aqua" and "options" in clean:
        clean["options"] = list(clean["options"])[:5]
    if dataset == "humaneval":
        clean["language"] = clean.get("language", "py")
        clean["stop_tokens"] = clean.get("stop_tokens") or DEFAULT_STOP_TOKENS
    return clean


def build_generation_prompt(
    dataset: str,
    examples: Sequence[Dict[str, Any]],
    batch_size: int,
) -> List[Dict[str, str]]:
    spec = DATASET_SPECS[dataset]
    schema = {key: "string" for key in spec["required"]}
    if dataset == "aqua":
        schema["options"] = ["A)...", "B)...", "C)...", "D)...", "E)..."]
    if dataset == "humaneval":
        schema.update({
            "language": "py",
            "stop_tokens": DEFAULT_STOP_TOKENS,
        })
    user_prompt = {
        "dataset": dataset,
        "task": f"Generate {batch_size} new unlabeled records.",
        "description": spec["description"],
        "output_schema": schema,
        "forbidden_fields": spec["forbidden"],
        "style_examples": list(examples),
        "requirements": [
            "Return a JSON array only. Do not wrap it in markdown.",
            "Generate novel records, not paraphrases of the examples.",
            "Match the difficulty, wording style, and required answer format of the examples.",
            "Do not include labels, answers, rationales, tests, or solutions.",
            "Avoid ambiguous or underspecified questions.",
        ],
    }
    if dataset in {"gsm8k", "svamp", "multiarith"}:
        user_prompt["requirements"].append(
            "The problem must be solvable with arithmetic reasoning and should have one final numeric answer."
        )
    if dataset == "mmlu":
        user_prompt["requirements"].append(
            "Options A-D must be plausible and mutually exclusive. Do not reveal which one is correct."
        )
    if dataset == "aqua":
        user_prompt["requirements"].append(
            "Options must be exactly five strings labeled A), B), C), D), and E)."
        )
    if dataset == "humaneval":
        user_prompt["requirements"].append(
            "The prompt must contain a Python function signature and docstring examples, but no hidden tests."
        )
    return [
        {
            "role": "system",
            "content": (
                "You generate unlabeled training questions for reasoning research. "
                "You must preserve the dataset schema exactly and never include answer labels."
            ),
        },
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False, indent=2)},
    ]


def parse_json_array(text: str) -> List[Dict[str, Any]]:
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(raw[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("Model output must be a JSON array.")
    return [item for item in data if isinstance(item, dict)]


def normalize_record(dataset: str, record: Dict[str, Any], index: int) -> Dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    clean: Dict[str, Any] = {}
    for key in spec["required"]:
        if key not in record:
            raise ValueError(f"Missing required field {key!r}: {record}")
        clean[key] = record[key]
    for key in spec["forbidden"]:
        clean.pop(key, None)
    if dataset == "aqua":
        options = clean["options"]
        if not isinstance(options, list) or len(options) != 5:
            raise ValueError(f"AQuA options must be a list of five choices: {record}")
        clean["options"] = [str(option).strip() for option in options]
    if dataset == "humaneval":
        clean["name"] = str(clean["name"]).strip() or f"SyntheticHumanEval_{index}"
        clean["language"] = "py"
        clean["stop_tokens"] = DEFAULT_STOP_TOKENS
    else:
        for key, value in list(clean.items()):
            if isinstance(value, str):
                clean[key] = value.strip()
    return clean


def dedupe_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for record in records:
        key = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


async def generate_batch(
    client: AsyncOpenAI,
    model: str,
    dataset: str,
    examples: Sequence[Dict[str, Any]],
    batch_size: int,
    max_tokens: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    messages = build_generation_prompt(dataset, examples, batch_size)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or "[]"
    return parse_json_array(content)


async def generate_dataset(args, dataset: str) -> None:
    spec = DATASET_SPECS[dataset]
    input_path = Path(args.input or spec["default_input"])
    train_limit = training_seed_limit(dataset, args)
    examples = load_seed_records(dataset, input_path, min(args.seed_examples, train_limit))
    if not examples:
        print(f"[skip] {dataset}: no seed records found at {input_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / spec["output_name"]
    print(
        f"[seed] {dataset}: using at most {len(examples)} records from the training seed region "
        f"of {input_path}"
    )

    base_url = args.base_url or os.getenv("AGENT_BASE_URL") or os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
    api_key = args.api_key or os.getenv("AGENT_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required for generation. Install it in the environment "
            "used to run this script, or run inside the project's configured venv."
        ) from exc
    client = AsyncOpenAI(**_client_kwargs(base_url, api_key))

    generated: List[Dict[str, Any]] = []
    failures = 0
    while len(generated) < args.num_records:
        remaining = args.num_records - len(generated)
        request_size = min(args.batch_size, remaining)
        try:
            raw_records = await generate_batch(
                client,
                args.model,
                dataset,
                random.sample(examples, min(len(examples), args.seed_examples_per_batch)),
                request_size,
                args.max_tokens,
                args.temperature,
            )
            normalized = [
                normalize_record(dataset, record, len(generated) + idx)
                for idx, record in enumerate(raw_records)
            ]
            generated = dedupe_records([*generated, *normalized])
            print(f"[{dataset}] {len(generated)}/{args.num_records}")
        except Exception as exc:
            failures += 1
            print(f"[warn] {dataset}: generation batch failed: {exc}")
            if failures >= args.max_failures:
                raise
            await asyncio.sleep(args.retry_sleep)

    write_records(dataset, output_path, generated[:args.num_records])
    print(f"[done] {dataset}: wrote {output_path}")


def write_records(dataset: str, path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fmt = DATASET_SPECS[dataset]["format"]
    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate unlabeled synthetic datasets with LLM prompts.")
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASET_SPECS.keys()],
        default="all",
        help="Dataset to generate.",
    )
    parser.add_argument("--input", type=str, default="", help="Optional seed input path for a single dataset.")
    parser.add_argument("--output_dir", type=str, default="ExpanData/unlabeled")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--num_records", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--seed_examples", type=int, default=64)
    parser.add_argument("--seed_examples_per_batch", type=int, default=6)
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help=(
            "Batch size used by the current training scripts. For non-MMLU datasets, "
            "only the first train_batch_size * train_num_iterations records are used as style seeds."
        ),
    )
    parser.add_argument(
        "--train_num_iterations",
        type=int,
        default=10,
        help=(
            "Number of training batches used by the current training scripts. For non-MMLU datasets, "
            "later records are treated as evaluation data and are not used as seeds."
        ),
    )
    parser.add_argument(
        "--seed_train_records",
        type=int,
        default=None,
        help=(
            "Explicit cap on the number of leading training records allowed as style seeds. "
            "Use this when your train/eval split differs from the default runner settings."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_failures", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    datasets = list(DATASET_SPECS.keys()) if args.dataset == "all" else [args.dataset]
    if args.input and len(datasets) != 1:
        raise ValueError("--input can only be used when --dataset is not all.")
    for dataset in datasets:
        await generate_dataset(args, dataset)


if __name__ == "__main__":
    asyncio.run(main())
