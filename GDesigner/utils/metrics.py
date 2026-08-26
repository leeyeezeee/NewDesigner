import json
from pathlib import Path
from typing import Any, Dict, Optional

from GDesigner.utils.globals import CompletionTokens, Cost, LLMCalls, PromptTokens


def reset_usage_counters() -> None:
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    LLMCalls.instance().reset()


def usage_snapshot() -> Dict[str, float]:
    """Return the usage counters at one exact point in the current process."""
    return {
        "cost": float(Cost.instance().value),
        "prompt_tokens": float(PromptTokens.instance().value),
        "completion_tokens": float(CompletionTokens.instance().value),
        "llm_calls": float(LLMCalls.instance().value),
    }


def usage_delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    """Subtract two snapshots, rejecting impossible negative counter deltas."""
    delta = {
        key: float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        for key in ("cost", "prompt_tokens", "completion_tokens", "llm_calls")
    }
    if any(value < 0.0 for value in delta.values()):
        raise ValueError("Usage counters were reset between snapshots.")
    return delta


def write_metrics_record(metrics_file: Optional[str], record: Dict[str, Any]) -> None:
    if not metrics_file:
        return

    output_path = Path(metrics_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **record,
        "cost": Cost.instance().value,
        "prompt_tokens": PromptTokens.instance().value,
        "completion_tokens": CompletionTokens.instance().value,
        "llm_calls": int(LLMCalls.instance().value),
    }
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
