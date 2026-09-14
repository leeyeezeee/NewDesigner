import asyncio
import unittest
from unittest.mock import patch

from GDesigner.llm.price import cost_count
from GDesigner.utils.metrics import reset_usage_counters, usage_snapshot
from GDesigner.utils.rollout_usage import collect_rollout_usage
from experiments.graph_concurrency import limited_graph_arun


def record_request(prompt_tokens, completion_tokens=1000):
    # Exercise the same accounting entry point as the remote LLM response path.
    cost_count("prompt", "answer", "test-model",
               prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class FakeGraph:
    def __init__(self, tokens, fail=False):
        self.tokens = tokens
        self.fail = fail

    async def arun(self):
        record_request(self.tokens)
        await asyncio.sleep(0)

        async def final_decision():
            await asyncio.sleep(0)
            record_request(7, 9000)

        await asyncio.create_task(final_decision())
        if self.fail:
            raise RuntimeError("failed graph")
        return ["answer"], 0


class RolloutUsageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.prices = patch.dict("os.environ", {
            "LOCAL_MODEL_INPUT_PRICE_PER_1K": "",
            "LOCAL_MODEL_OUTPUT_PRICE_PER_1K": "",
        })
        self.prices.start()
        self.addCleanup(self.prices.stop)
        reset_usage_counters()

    async def test_parallel_graphs_and_reward_calls_are_isolated(self):
        for limit in [None, 1, 2]:
            with self.subTest(limit=limit):
                reset_usage_counters()
                graphs = [FakeGraph(100), FakeGraph(200)]
                semaphore = asyncio.Semaphore(limit) if limit else None
                results = await asyncio.gather(*[
                    limited_graph_arun(semaphore, graph) for graph in graphs
                ])
                self.assertEqual(results, [(["answer"], 0), (["answer"], 0)])
                self.assertEqual([g.rollout_prompt_tokens for g in graphs], [107, 207])
                # TF / counterfactual scoring occurs after the rollout scope.
                record_request(50000)
                self.assertEqual([g.rollout_prompt_tokens for g in graphs], [107, 207])
                self.assertEqual(usage_snapshot()["prompt_tokens"], 50314)
                self.assertEqual(usage_snapshot()["completion_tokens"], 21000)
                self.assertEqual(usage_snapshot()["llm_calls"], 5)

    async def test_exception_and_cancellation_restore_context(self):
        with collect_rollout_usage() as outer:
            broken = FakeGraph(100, fail=True)
            with self.assertRaises(RuntimeError):
                await limited_graph_arun(None, broken)
            self.assertIsNone(broken.rollout_prompt_tokens)
            record_request(3)
            self.assertEqual(outer.prompt_tokens, 3)

        started = asyncio.Event()

        class CanceledGraph:
            async def arun(self):
                record_request(10)
                started.set()
                await asyncio.Event().wait()

        canceled = CanceledGraph()
        task = asyncio.create_task(limited_graph_arun(None, canceled))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(canceled.rollout_prompt_tokens)
        good = FakeGraph(20)
        await limited_graph_arun(None, good)
        self.assertEqual(good.rollout_prompt_tokens, 27)


if __name__ == "__main__":
    unittest.main()
