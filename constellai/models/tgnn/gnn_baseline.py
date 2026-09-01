"""Step-3 (part A) on the M3 ladder: static-GNN conjunction classifier.

Isolates the value of graph structure BEFORE adding temporal modeling --
per the project roadmap, this must exist and be evaluated before the
full continuous-time TGNN, so a later TGNN-vs-LSTM win can be attributed
to graph structure, temporal modeling, or both, rather than guessed at.

Deliberately plain PyTorch, no PyG/DGL -- one mean-aggregation GCN layer
is enough to test whether neighbor information helps at all; adding a
graph library is a decision for once this baseline has a number to beat.
"""

from __future__ import annotations

import torch
from torch import nn


class GCNLayer(nn.Module):
    """One mean-aggregation graph-conv step: each node's new embedding
    is a linear function of its own features plus the mean of its
    neighbors' features (self-loop included)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = node_features.size(0)
        agg = node_features.clone()
        counts = torch.ones(n, device=node_features.device)

        if edge_index.numel() > 0:
            src, dst = edge_index[:, 0], edge_index[:, 1]
            agg.index_add_(0, dst, node_features[src])
            agg.index_add_(0, src, node_features[dst])
            counts.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            counts.index_add_(0, src, torch.ones_like(src, dtype=torch.float))

        agg = agg / counts.unsqueeze(1)
        return torch.relu(self.linear(agg))


class EdgeRiskGNN(nn.Module):
    """Two GCN layers to get node embeddings, then an MLP over
    [embedding_a, embedding_b, raw edge features] to predict edge risk."""

    def __init__(self, node_in_dim: int = 4, edge_in_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.gcn1 = GCNLayer(node_in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, edge_index, edge_features) -> torch.Tensor:
        h = self.gcn1(node_features, edge_index)
        h = self.gcn2(h, edge_index)

        if edge_index.numel() == 0:
            return torch.zeros(0)

        src, dst = edge_index[:, 0], edge_index[:, 1]
        edge_input = torch.cat([h[src], h[dst], edge_features], dim=1)
        return self.edge_head(edge_input).squeeze(-1)


def train_one_epoch(model, snapshot, optimizer, pos_weight=None, device="cpu"):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    node_features = torch.from_numpy(snapshot.node_features).to(device)
    edge_index = torch.from_numpy(snapshot.edge_index).to(device)
    edge_features = torch.from_numpy(snapshot.edge_features).to(device)
    labels = torch.from_numpy(snapshot.edge_labels).float().to(device)

    optimizer.zero_grad()
    logits = model(node_features, edge_index, edge_features)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()