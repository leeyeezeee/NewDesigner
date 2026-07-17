from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def resolve_edge_training_log_file(dataset: str) -> Path:
    return Path("result") / f"{dataset}_log.jsonl"


def reset_edge_training_log(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("", encoding="utf-8")


def resolve_question_id(record: Any, fallback_index: int) -> Any:
    for field in ("question_id", "task_id", "problem_id", "id"):
        try:
            value = record.get(field)
        except (AttributeError, TypeError):
            value = None
        if value is not None and str(value).strip():
            return str(value)
    record_name = getattr(record, "name", None)
    if record_name is not None:
        return str(record_name)
    return str(int(fallback_index))


def append_edge_training_details(
    log_file: Path,
    *,
    question_id: Any,
    edge_details: Dict[str, Dict[str, Any]],
) -> None:
    if not edge_details:
        return
    edges = []
    for detail in edge_details.values():
        if detail.get("type") != "spatial" or "ig_gain" not in detail:
            continue
        edge_id = f"{detail.get('source')}->{detail.get('target')}"
        edges.append({
            "edge": edge_id,
            "round_id": int(detail.get("round", 0)),
            "ig_gain": float(detail["ig_gain"]),
            "edge_token_cost": float(detail.get("edge_token_cost", 0.0)),
        })
    if not edges:
        return
    with log_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps({
            "question_id": question_id,
            "edges": edges,
        }, ensure_ascii=False) + "\n")
