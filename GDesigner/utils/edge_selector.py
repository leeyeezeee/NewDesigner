import random
from collections import deque
from typing import Dict, Iterable, List

import torch
from torch import nn


class EdgeSelector(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class SelectorReplayBuffer:
    def __init__(self, capacity: int = 512):
        self.samples = deque(maxlen=max(1, int(capacity)))

    def __len__(self) -> int:
        return len(self.samples)

    def add_many(self, samples: Iterable[Dict[str, torch.Tensor]]) -> None:
        for sample in samples:
            self.samples.append(sample)

    def sample(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        batch_size = min(max(1, int(batch_size)), len(self.samples))
        return random.sample(list(self.samples), batch_size)


def build_edge_selector_examples(
    graph,
    task: str,
    edge_details: Dict[str, Dict],
    entropy_tau: float,
    ig_tau: float = 0.0,
) -> List[Dict[str, torch.Tensor]]:
    if not edge_details:
        return []

    task_embedding = graph.edge_selector_task_embedding(task)
    examples: List[Dict[str, torch.Tensor]] = []
    for detail in edge_details.values():
        source_id = detail["source"]
        target_id = detail["target"]
        if source_id not in graph.nodes or target_id not in graph.nodes:
            continue
        features = graph.edge_selector_feature(
            task,
            source_id,
            target_id,
            task_embedding=task_embedding,
        )
        uncertainty_delta = float(detail.get("uncertainty_delta", detail.get("entropy_delta", 0.0)))
        ig_gain = detail.get("ig_gain")
        if ig_gain is None:
            label = 1.0 if uncertainty_delta > entropy_tau else 0.0
        else:
            ig_gain = float(ig_gain)
            label = 1.0 if (
                ig_gain > ig_tau
                or (ig_gain >= 0.0 and uncertainty_delta > entropy_tau)
            ) else 0.0
        examples.append({
            "features": features.detach().float(),
            "label": torch.tensor(label, dtype=torch.float32),
        })
    return examples


def train_edge_selector(
    selector: EdgeSelector,
    optimizer: torch.optim.Optimizer,
    replay_buffer: SelectorReplayBuffer,
    batch_size: int = 32,
    train_steps: int = 3,
) -> bool:
    if len(replay_buffer) < max(1, int(batch_size)):
        return False

    selector.train()
    loss_fn = nn.BCEWithLogitsLoss()
    device = next(selector.parameters()).device
    for _ in range(max(1, int(train_steps))):
        batch = replay_buffer.sample(batch_size)
        features = torch.stack([sample["features"] for sample in batch]).to(device)
        labels = torch.stack([sample["label"] for sample in batch]).to(device)
        loss = loss_fn(selector(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return True
