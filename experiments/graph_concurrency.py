import asyncio
from typing import Any

from GDesigner.llm.price import track_graph_token_usage


def make_graph_semaphore(max_concurrent_graphs: int | None) -> asyncio.Semaphore | None:
    if max_concurrent_graphs is None:
        return None
    max_concurrent_graphs = int(max_concurrent_graphs)
    if max_concurrent_graphs <= 0:
        return None
    return asyncio.Semaphore(max_concurrent_graphs)


async def limited_graph_arun(
    semaphore: asyncio.Semaphore | None,
    realized_graph,
    *args: Any,
    track_graph_tokens: bool = False,
    **kwargs: Any,
):
    async def execute_graph():
        if not track_graph_tokens:
            return await realized_graph.arun(*args, **kwargs)
        with track_graph_token_usage() as usage:
            try:
                return await realized_graph.arun(*args, **kwargs)
            finally:
                realized_graph.graph_token_usage = dict(usage)

    if semaphore is None:
        return await execute_graph()
    async with semaphore:
        return await execute_graph()


async def limited_async_call(
    semaphore: asyncio.Semaphore | None,
    async_fn,
    *args: Any,
    **kwargs: Any,
):
    if semaphore is None:
        return await async_fn(*args, **kwargs)
    async with semaphore:
        return await async_fn(*args, **kwargs)
