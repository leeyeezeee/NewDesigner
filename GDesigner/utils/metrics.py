import json
from pathlib import Path
from typing import Any, Dict, Optional

from GDesigner.utils.globals import CompletionTokens, Cost, PromptTokens


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
    }
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
