import asyncio
from typing import Any

from GDesigner.utils.rollout_usage import collect_rollout_usage

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
    async def execute_graph():
        # Scope ends before TF/edge-ablation scoring. Include the regular final
        # decision call, but never another concurrently executing graph's usage.
        realized_graph.rollout_prompt_tokens = None
        with collect_rollout_usage() as usage:
            result = await realized_graph.arun(*args, **kwargs)
        realized_graph.rollout_prompt_tokens = usage.prompt_tokens
        return result

    if semaphore is None:
        return await execute_graph()
    async with semaphore:
        return await execute_graph()
