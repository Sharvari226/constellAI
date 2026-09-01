"""M3 Step 3: minimal continuous-time temporal graph model ("TGN-lite").

Each satellite has a persistent memory vector that updates via a GRU
cell every time a new event involving it occurs, chronologically -- so
a node's representation is shaped by ALL its neighbors' interactions
over time, not just one pair's trajectory (LSTM's limitation) and not
just one static snapshot (the static-GNN baseline's original
limitation). This is a deliberately small version of TGN's memory
module (Rossi et al. 2020) -- no separate message/aggregator/embedding
submodules yet, just memory + a link-risk head.
"""

from __future__ import annotations

import torch
from torch import nn

from constellai.models.tgnn.dynamic_graph import DynamicGraphData


class TGNLite(nn.Module):
    def __init__(self, feature_dim: int = 4, memory_dim: int = 16):
        super().__init__()
        self.memory_dim = memory_dim
        self.message = nn.Linear(feature_dim, memory_dim)
        self.memory_update = nn.GRUCell(memory_dim, memory_dim)
        self.risk_head = nn.Sequential(
            nn.Linear(2 * memory_dim + feature_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )

    def run_events(self, num_nodes: int, events, device="cpu") -> torch.Tensor:
        """Process the chronological event stream, updating per-node
        memory as it goes. Returns final memory, shape (num_nodes, memory_dim).

        Memory is kept as a list of tensors, not one indexed tensor --
        indexed in-place writes into a single tensor break autograd once
        the same node is updated more than once (which happens constantly
        here), since each write invalidates the graph the previous write
        depends on.

        Node A and node B get DIFFERENT messages, not the same one --
        features are a relative vector (A's position minus B's), so from
        B's side that vector points the opposite way. Feeding both nodes
        the identical message discarded this directionality entirely.
        """
        memory = [torch.zeros(1, self.memory_dim, device=device) for _ in range(num_nodes)]

        for event in events:
            feat = torch.from_numpy(event.features).unsqueeze(0).to(device)
            feat_from_b = feat.clone()
            feat_from_b[:, :3] *= -1

            msg_a = self.message(feat)
            msg_b = self.message(feat_from_b)
            memory[event.node_a] = self.memory_update(msg_a, memory[event.node_a])
            memory[event.node_b] = self.memory_update(msg_b, memory[event.node_b])

        return torch.cat(memory, dim=0)

    def predict_pair(self, memory: torch.Tensor, node_a: int, node_b: int, last_features: torch.Tensor) -> torch.Tensor:
        pair_input = torch.cat([memory[node_a], memory[node_b], last_features])
        return self.risk_head(pair_input.unsqueeze(0)).squeeze()


def forward_and_score(model: TGNLite, graph: DynamicGraphData, device="cpu"):
    """Runs the full event stream once, then scores every pair that has
    a label, using each pair's last-seen features + final memory."""
    num_nodes = len(graph.node_ids)
    memory = model.run_events(num_nodes, graph.events, device=device)

    last_features: dict[tuple[int, int], torch.Tensor] = {}
    for event in graph.events:
        last_features[(event.node_a, event.node_b)] = torch.from_numpy(event.features).to(device)

    logits, pairs, labels = [], [], []
    for pair, label in graph.pair_labels.items():
        if pair not in last_features:
            continue
        logits.append(model.predict_pair(memory, pair[0], pair[1], last_features[pair]))
        pairs.append(pair)
        labels.append(label)

    return torch.stack(logits) if logits else torch.zeros(0), labels