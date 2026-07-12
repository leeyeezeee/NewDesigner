"""Write uncaught experiment failures to the process working directory."""

from __future__ import annotations

import asyncio
import os
import platform
from datetime import datetime
from pathlib import Path
import sys
import traceback
from typing import Awaitable, Callable, TypeVar


T = TypeVar("T")


def _write_crash_log(exc: BaseException) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    script_name = Path(sys.argv[0]).stem or "experiment"
    log_path = Path.cwd() / f"crash_{script_name}_{timestamp}_{os.getpid()}.log"
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
