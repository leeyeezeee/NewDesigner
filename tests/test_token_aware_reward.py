import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from experiments.edge_training_log import append_training_step
from experiments.teacher_forcing_reward import (
    _centered_advantages,
    _standardized_advantages,
    add_teacher_forcing_reward_args,
    experiment_summary_metadata,
    graph_correctness_advantage_edge_loss,
    resolve_graph_reward_sampling,
    token_aware_graph_utilities,
)


def sampled_graph(parameter, actions, tokens, gains):
    distribution = torch.distributions.Bernoulli(logits=parameter)
    log_probs = distribution.log_prob(parameter.new_tensor(actions))
    edges, details = [], {}
    for i, (action, gain) in enumerate(zip(actions, gains)):
        if action:
            key = f"spatial:0:{i}->{i + 1}"
            edges.append(dict(
                type="spatial", round=0, source=str(i), target=str(i + 1),
                edge_key=key, log_prob=log_probs[i],
            ))
            details[key] = {"ig_gain": gain}
    graph = SimpleNamespace(
        rollout_prompt_tokens=tokens, edge_log_probs=edges,
        mean_spatial_edges_per_round=sum(actions),
        spatial_parameters=lambda: [parameter], temporal_logits=None,
    )
    return graph, log_probs, details


class TokenAwareRewardTests(unittest.TestCase):
    def test_small_cost_differences_are_not_standardized(self):
        _, utilities = token_aware_graph_utilities([1, 1], [10000, 10010], 0.1)
        _, advantages, _, _ = _centered_advantages(utilities)
        self.assertAlmostEqual(advantages[0], 0.1 * 10 / 10010 / 2)
        self.assertAlmostEqual(advantages[1], -advantages[0])
        self.assertLess(abs(advantages[0]), 0.0001)
        _, smaller = token_aware_graph_utilities([1, 1], [10000, 10010], 0.01)
        self.assertAlmostEqual(_centered_advantages(smaller)[1][0], advantages[0] / 10)

    def test_correctness_order_zero_cost_and_zero_beta(self):
        _, utilities = token_aware_graph_utilities([1, 0], [10000, 1], 0.1)
        self.assertGreater(utilities[0], utilities[1])
        self.assertEqual(token_aware_graph_utilities([1, 0], [0, 0], 0.1),
                         ([0.0, 0.0], [1.0, 0.0]))
        self.assertEqual(token_aware_graph_utilities([1, 0], [100, 200], 0)[1], [1.0, 0.0])

    def test_invalid_cost_inputs_fail(self):
        for beta in [-1, float("nan"), float("inf")]:
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                token_aware_graph_utilities([1], [1], beta)
        for tokens in [[-1], [float("nan")], [float("inf")], []]:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                token_aware_graph_utilities([1], tokens, 0.1)

    def test_defaults_sampling_and_metadata(self):
        parser = argparse.ArgumentParser()
        add_teacher_forcing_reward_args(parser)
        args = parser.parse_args([])
        self.assertEqual(args.prompt_token_cost_beta, 0.1)
        self.assertTrue(resolve_graph_reward_sampling(False, 0, 0.1, 8))
        self.assertFalse(resolve_graph_reward_sampling(False, 0, 0, 1))
        self.assertTrue(resolve_graph_reward_sampling(True, 0, 0, 8))
        with self.assertRaises(ValueError):
            resolve_graph_reward_sampling(False, 0, 0.1, 1)
        reward = experiment_summary_metadata(args, "mmlu")["reward"]
        self.assertEqual(reward["graph"], "centered_correctness_minus_prompt_token_cost")
        self.assertEqual(reward["prompt_token_cost_beta"], 0.1)

    def test_graph_and_edge_objectives_have_identical_gradients(self):
        theta = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.6], dtype=torch.float64))
        first = sampled_graph(theta, [1, 0, 1], 100, [0.8, 0, -0.2])
        second = sampled_graph(theta, [0, 1, 1], 200, [0, 0.3, -0.4])
        graphs, per_edge_log_probs, details = map(list, zip(first, second))
        graph_log_probs = [value.sum() for value in per_edge_log_probs]
        loss, summaries = graph_correctness_advantage_edge_loss(
            [graphs], [graph_log_probs], [[1, 0]], [details], theta.sum(),
            prompt_token_cost_beta=0.1, edge_ig_reward_lambda=0.7,
            edge_ig_discount_factor=0,
        )
        # u = [0.95, -0.1]; centered utility = [+0.525, -0.525].
        # Graph coefficients multiply ALL sampled actions, not only selected edges.
        expected = theta.new_tensor(0.0)
        for advantage, graph, log_probs, edge_details in zip(
            [0.525, -0.525], graphs, per_edge_log_probs, details
        ):
            expected -= advantage * log_probs.sum() / 2
            for edge in graph.edge_log_probs:
                coefficient = 0.7 * math.tanh(edge_details[edge["edge_key"]]["ig_gain"])
                expected -= coefficient * edge["log_prob"] / 2
        torch.testing.assert_close(loss, expected)
        actual_gradient = torch.autograd.grad(loss, theta, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, theta)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertEqual(summaries[0]["rollout_prompt_tokens"], [100, 200])
        self.assertAlmostEqual(summaries[0]["token_aware_advantages"][0], 0.525)
        norms = summaries[0]["batch_gradient_norms"]
        self.assertGreater(norms["correctness"], 0)
        self.assertGreater(norms["prompt_token_cost"], 0)
        self.assertGreater(norms["edge_ig"], 0)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "steps.jsonl"
            append_training_step(path, step=0, accuracy=0.5, avg_edges=2,
                                 avg_communication_tokens=180,
                                 graph_reward_summaries=summaries)
            record = json.loads(path.read_text())
            self.assertEqual(record["graph_rewards"][0]["rollout_prompt_tokens"], [100, 200])
            self.assertNotIn("edge_ig_records", record["graph_rewards"][0])

    def test_more_edges_do_not_add_an_ig_only_scale_factor(self):
        ratios = []
        for count in [1, 20]:
            theta = torch.nn.Parameter(torch.zeros(count, dtype=torch.float64))
            first = sampled_graph(theta, [1] * count, 100, [0.5] * count)
            second = sampled_graph(theta, [0] * count, 100, [0] * count)
            graphs, log_probs, details = map(list, zip(first, second))
            _, summaries = graph_correctness_advantage_edge_loss(
                [graphs], [[x.sum() for x in log_probs]], [[1, 0]], [details],
                theta.sum(), prompt_token_cost_beta=0.1,
                edge_ig_reward_lambda=1, edge_ig_discount_factor=0,
            )
            norms = summaries[0]["batch_gradient_norms"]
            ratios.append(norms["edge_ig"] / norms["correctness"])
        self.assertAlmostEqual(ratios[0], ratios[1], places=6)

    def test_full_graph_tf_standardization_is_unchanged(self):
        theta = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        first = sampled_graph(theta, [1], 100, [0])
        second = sampled_graph(theta, [0], 100, [0])
        graphs, log_probs, details = map(list, zip(first, second))
        loss, summaries = graph_correctness_advantage_edge_loss(
            [graphs], [[x.sum() for x in log_probs]], [[1, 1]], [details],
            theta.sum(), graph_tf_score_groups=[[-1000, -1002]],
            full_graph_tf_reward_lambda=0.7,
        )
        self.assertEqual(summaries[0]["full_graph_tf_advantages"], [1.0, -1.0])
        self.assertEqual(summaries[0]["token_aware_advantages"], [0.0, 0.0])
        expected = -0.7 * (log_probs[0].sum() - log_probs[1].sum()) / 2
        torch.testing.assert_close(torch.autograd.grad(loss, theta, retain_graph=True)[0],
                                   torch.autograd.grad(expected, theta)[0])
        self.assertEqual(_standardized_advantages([1, 1], advantage_epsilon=1e-6)[1], [0, 0])

    def test_cost_normalizes_per_question_and_requires_accounting(self):
        theta = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        groups = [
            [sampled_graph(theta, [1], t, [0]) for t in tokens]
            for tokens in [[100, 200], [1000, 2000]]
        ]
        graph_groups = [[sample[0] for sample in group] for group in groups]
        log_probs = [[sample[1].sum() for sample in group] for group in groups]
        details = [[sample[2] for sample in group] for group in groups]
        _, summaries = graph_correctness_advantage_edge_loss(
            graph_groups, log_probs, [[1, 1], [1, 1]], details, theta.sum(),
        )
        self.assertEqual(summaries[0]["token_aware_advantages"],
                         summaries[1]["token_aware_advantages"])
        graph_groups[0][0].rollout_prompt_tokens = None
        with self.assertRaisesRegex(ValueError, "accounting"):
            graph_correctness_advantage_edge_loss(
                graph_groups, log_probs, [[1, 1], [1, 1]], details, theta.sum(),
            )


if __name__ == "__main__":
    unittest.main()
