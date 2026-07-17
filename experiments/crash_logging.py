"""Write uncaught experiment failures to the process working directory."""

from __future__ import annotations

import asyncio
import os
import platform
import re
from datetime import datetime
from pathlib import Path
import sys
import traceback
from typing import Awaitable, Callable, TypeVar


T = TypeVar("T")


def _argument_value(name: str) -> str | None:
    prefix = f"{name}="
    for index, argument in enumerate(sys.argv[1:]):
        if argument.startswith(prefix):
            return argument[len(prefix):]
        if argument == name and index + 2 < len(sys.argv):
            return sys.argv[index + 2]
    return None


def _dataset_name() -> str:
    """Resolve a stable dataset name for the current experiment entry point."""
    explicit_name = _argument_value("--dataset") or _argument_value("--domain")
    if explicit_name:
        dataset_name = explicit_name
    else:
        dataset_name = Path(sys.argv[0]).stem or "experiment"
        for prefix in ("run_", "train_", "evaluate_"):
            if dataset_name.startswith(prefix):
                dataset_name = dataset_name[len(prefix):]
                break

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name).strip("._-")
    return safe_name.lower() or "experiment"


def _write_crash_log(exc: BaseException) -> Path:
    # A later failure for the same dataset intentionally replaces the previous
    # one so the working directory contains only the latest relevant traceback.
    log_path = Path.cwd() / f"{_dataset_name()}_error.log"
    traceback_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    log_path.write_text(
        "\n".join(
            [
                f"timestamp: {datetime.now().astimezone().isoformat()}",
                f"pid: {os.getpid()}",
                f"cwd: {Path.cwd()}",
                f"python: {sys.version}",
                f"platform: {platform.platform()}",
                f"command: {' '.join(sys.argv)}",
                "",
                "uncaught exception:",
                traceback_text,
            ]
        ),
        encoding="utf-8",
    )
    return log_path


def run_async_with_crash_logging(
    async_main: Callable[[], Awaitable[T]],
) -> T:
    """Run an async entry point and persist any fatal uncaught exception."""
    try:
        return asyncio.run(async_main())
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        try:
            log_path = _write_crash_log(exc)
            print(f"Fatal error log written to: {log_path}", file=sys.stderr)
        except Exception as log_exc:
            print(
                f"Failed to write fatal error log: {log_exc!r}",
                file=sys.stderr,
            )
        raise
