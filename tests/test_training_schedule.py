import ast
import json
import unittest
from pathlib import Path

from experiments.training_schedule import fixed_split_schedule


ROOT = Path(__file__).resolve().parents[1]


class TrainingScheduleTests(unittest.TestCase):
    def test_actual_dataset_samples_match_historical_ten_batch_split(self):
        paths = [
            'datasets/AQuA/AQuA.jsonl',
            'datasets/gsm8k/gsm8k.jsonl',
            'datasets/humaneval/humaneval-py.jsonl',
            'datasets/MultiArith/MultiArith.json',
            'datasets/SVAMP/SVAMP.json',
        ]
        for relative in paths:
            path = ROOT / relative
            with self.subTest(dataset=relative):
                content = path.read_text(encoding='utf-8')
                records = ([json.loads(line) for line in content.splitlines() if line.strip()]
                           if path.suffix == '.jsonl' else json.loads(content))
                size = len(records)
                historical_train = list(range(40))
                historical_eval = list(range(40, size // 4 * 4))
                for updates in [10, 30, 100]:
                    schedule = fixed_split_schedule(size, 4, updates)
                    train_ids = [i for batch, train in schedule if train
                                 for i in range(batch * 4, (batch + 1) * 4)]
                    eval_ids = [i for batch, train in schedule if not train
                                for i in range(batch * 4, (batch + 1) * 4)]
                    self.assertEqual(sorted(set(train_ids)), historical_train)
                    self.assertEqual(eval_ids, historical_eval)
                    self.assertTrue(set(train_ids).isdisjoint(eval_ids))
                    self.assertEqual(sum(train for _, train in schedule), updates)
                    if updates == 30:
                        self.assertEqual(train_ids, historical_train * 3)
                    self.assertEqual([records[i] for i in eval_ids], [records[i] for i in historical_eval])

    def test_short_training_and_fixed_baseline_keep_evaluation_suffix(self):
        for updates, enabled in [(0, True), (3, True), (30, False)]:
            schedule = fixed_split_schedule(254, 4, updates, optimize_enabled=enabled)
            self.assertEqual([idx for idx, train in schedule if not train], list(range(10, 63)))
            self.assertEqual(sum(train for _, train in schedule), updates if enabled else 0)
            # The training-completion boundary is reached exactly once, or never.
            finishes = [step for step, (_, train) in enumerate(schedule)
                        if train and step + 1 == updates]
            self.assertEqual(len(finishes), int(enabled and updates > 0))

    def test_explicit_legacy_split_and_invalid_budgets(self):
        schedule = fixed_split_schedule(100, 4, 30, train_split_batches=5)
        self.assertEqual([idx for idx, train in schedule if train], list(range(5)) * 6)
        self.assertEqual([idx for idx, train in schedule if not train], list(range(5, 25)))
        for size, batch, updates, split in [(40,4,30,10), (100,0,30,10), (100,4,-1,10), (100,4,30,0)]:
            with self.assertRaises(ValueError):
                fixed_split_schedule(size, batch, updates, train_split_batches=split)

    def test_runners_slice_and_identify_records_by_dataset_index(self):
        # Check the integration points, not just the standalone schedule: using
        # the optimizer step for either slicing or fallback IDs leaks eval data.
        for name in ['run_gsm8k.py', 'run_humaneval.py', 'math_dataset_runner.py']:
            tree = ast.parse((ROOT/'experiments'/name).read_text(encoding='utf-8'))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            loaders = [n for n in calls if n.func.id == 'dataloader']
            self.assertEqual(len(loaders), 1, name)
            self.assertEqual(ast.unparse(loaders[0].args[-1]), 'dataset_batch_idx', name)
            identifiers = [n for n in calls if n.func.id == 'resolve_question_id']
            self.assertEqual(ast.unparse(identifiers[0].args[-1]), 'dataset_batch_idx * args.batch_size + i_record', name)
        tree = ast.parse((ROOT/'experiments/run_mmlu.py').read_text(encoding='utf-8'))
        splits = [n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Name) and n.func.id == 'MMLUDataset']
        self.assertEqual(splits, ['dev', 'val'])


if __name__ == '__main__':
    unittest.main()
