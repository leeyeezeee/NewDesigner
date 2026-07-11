import asyncio
from typing import Any


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
    **kwargs: Any,
):
    if semaphore is None:
        return await realized_graph.arun(*args, **kwargs)
    async with semaphore:
        return await realized_graph.arun(*args, **kwargs)


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
