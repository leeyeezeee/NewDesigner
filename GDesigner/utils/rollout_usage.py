"""Prompt-token accounting scoped to one asynchronous graph execution."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class RolloutUsage:
    prompt_tokens: int = 0


_rollout_usage: ContextVar[Optional[RolloutUsage]] = ContextVar(
    "rollout_usage", default=None
)


@contextmanager
def collect_rollout_usage() -> Iterator[RolloutUsage]:
    # Child tasks inherit this collector, while concurrent graphs get their own.
    usage = RolloutUsage()
    token = _rollout_usage.set(usage)
    try:
        yield usage
    finally:
        _rollout_usage.reset(token)


def record_rollout_prompt_tokens(prompt_tokens: int) -> None:
    usage = _rollout_usage.get()
    if usage is not None:
        usage.prompt_tokens += prompt_tokens
