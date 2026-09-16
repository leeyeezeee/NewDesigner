"""Plot measured paired attack summaries; accepts the same schema from other repos."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_groups(paths):
    groups = defaultdict(list)
    seen = set()
    reference = None
    for path in paths:
        row = json.loads(path.read_text(encoding='utf8'))
        key = (row['method'], row['seed'])
        if key in seen:
            raise ValueError(f'Duplicate method/seed: {key}')
        seen.add(key)
        if row['num_questions'] != len(row['question_indices']):
            raise ValueError(f'Incomplete evaluation: {path}')
        protocol = tuple(json.dumps(row[field], sort_keys=True) for field in [
            'dataset', 'dataset_sha256', 'question_indices', 'attack', 'attack_prompt',
            'llm_name', 'num_rounds', 'protocol', 'paired_topology',
        ])
        if reference is not None and protocol != reference:
            raise ValueError('Cannot compare different datasets, subsets, models, rounds or attacks.')
        reference = protocol
        if not all(0 <= row[field] <= 1 for field in ['clean_accuracy', 'attack_accuracy']):
            raise ValueError('Accuracies must be fractions in [0, 1].')
        groups[row['method']].append(row)
    if not groups:
        raise ValueError('No measured *.summary.json files found; run evaluations first.')
    seed_sets = [{row['seed'] for row in rows} for rows in groups.values()]
    if any(seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError('All methods must have the same evaluated seed set.')
    return groups


def plot(groups, output, order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    names = [name for name in order if name in groups]
    names += [name for name in groups if name not in names]
    means = [[100 * np.mean([row[field] for row in groups[name]]) for name in names]
             for field in ['clean_accuracy', 'attack_accuracy']]
    count = len(groups[names[0]])
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(max(6.0, len(names) * 1.35), 3.1))
        positions = np.arange(len(names))
        ax.bar(positions - .17, means[0], width=.34, color='#EEDDA5', label='Before attack')
        ax.bar(positions + .17, means[1], width=.34, color='#76AABB', label='After attack')
        ax.set_xticks(positions, names)
        ax.set_ylabel('Accuracy (%)')
        # Like the reference, allow a cropped accuracy axis, visibly labelled.
        lower = max(0, 5 * np.floor((min(min(values) for values in means) - 5) / 5))
        upper = min(100, 5 * np.ceil((max(max(values) for values in means) + 3) / 5))
        ax.set_ylim(lower, upper)
        ax.legend(loc='upper left', frameon=True)
        ax.tick_params(axis='x', length=0, labelsize=10)
        fig.text(.5, .025, f'Robustness under attack on AQuA ({count} seed' + ('s)' if count > 1 else ')'),
                 ha='center', fontfamily='DejaVu Serif')
        fig.tight_layout(rect=(0, .08, 1, 1))
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ['.pdf', '.png']:
            fig.savefig(output.with_suffix(suffix), dpi=300)
        plt.close(fig)
    return names, means


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('result/aqua_attack'))
    parser.add_argument('--output', type=Path, default=Path('result/aqua_attack/robustness_aqua'))
    parser.add_argument('--order', nargs='+', default=['Complete', 'Random', 'Tree', 'AgentPrune', 'ARG-Designer', 'EdgeIG'])
    args = parser.parse_args()
    plot(load_groups(sorted(args.results.glob('*.summary.json'))), args.output, args.order)
