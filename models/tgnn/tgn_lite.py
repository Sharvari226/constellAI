"""M3 Step 3: minimal continuous-time temporal graph model ("TGN-lite"),
with calibrated uncertainty AND per-pair memory snapshots.

Memory is DETACHED before each update (truncated backpropagation
through time) -- without this, backprop must traverse the entire prior
interaction history, which caused multi-hour training times before
this was fixed.

PER-PAIR MEMORY SNAPSHOT (this fix): run_events previously returned
only ONE final memory state, used to score every pair regardless of
when that pair's relevant interaction actually happened. With ~7,800
events per scenario, a satellite's memory gets overwritten many times
by OTHER interactions before the "final" state is read -- so scoring
pair (A,B) with final memory could be reading memory that's mostly
about A's later interactions with C, D, E, not about B at all. This
was a real information-loss bug, not just an under-training symptom.
The fix: track, per pair, a memory snapshot captured at THAT PAIR's
own most recent event (overwritten each time the pair recurs --
chronological processing means the last write is exactly the correct
snapshot). Scoring now reads each pair's own snapshot, not one shared
global final state.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Beta

from constellai.models.tgnn.dynamic_graph import DynamicGraphData


class TGNLite(nn.Module):
    def __init__(self, feature_dim: int = 5, memory_dim: int = 16):
        super().__init__()
        self.memory_dim = memory_dim
        self.message = nn.Linear(feature_dim, memory_dim)
        self.memory_update = nn.GRUCell(memory_dim, memory_dim)
        self.risk_head = nn.Sequential(
            nn.Linear(2 * memory_dim + feature_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 2),
        )

    def run_events(self, num_nodes: int, events, device="cpu"):
        """Process the chronological event stream.

        Returns
        -------
        (final_memory, pair_memory_snapshots)
            final_memory : torch.Tensor, shape (num_nodes, memory_dim)
                Memory after ALL events -- kept for callers that
                genuinely want "current global state" (e.g. a
                single-event query where there's nothing else to
                snapshot against).
            pair_memory_snapshots : dict[(node_a, node_b), (mem_a, mem_b)]
                Each pair's memory AT THE TIME of its own last event --
                this, not final_memory, is what scoring should use.
        """
        memory = [torch.zeros(1, self.memory_dim, device=device) for _ in range(num_nodes)]
        pair_memory_snapshots: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        for event in events:
            feat = torch.from_numpy(event.features).unsqueeze(0).to(device)
            feat_from_b = feat.clone()
            feat_from_b[:, :3] *= -1  # position components only; separation and rel_speed are unsigned magnitudes

            msg_a = self.message(feat)
            msg_b = self.message(feat_from_b)
            memory[event.node_a] = self.memory_update(msg_a, memory[event.node_a].detach())
            memory[event.node_b] = self.memory_update(msg_b, memory[event.node_b].detach())

            # Overwritten each time this pair recurs -- the last write,
            # by chronological processing order, is exactly this pair's
            # own most recent snapshot.
            pair_memory_snapshots[(event.node_a, event.node_b)] = (
                memory[event.node_a], memory[event.node_b]
            )

        return torch.cat(memory, dim=0), pair_memory_snapshots

    def predict_pair_distribution(
        self, mem_a: torch.Tensor, mem_b: torch.Tensor, last_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """mem_a, mem_b: (1, memory_dim) each -- typically the pair's OWN
        last-event snapshot from pair_memory_snapshots, not a global
        final memory tensor indexed by node id (that indexing approach
        is exactly what caused the stale-memory bug)."""
        pair_input = torch.cat([mem_a.squeeze(0), mem_b.squeeze(0), last_features])
        raw = self.risk_head(pair_input.unsqueeze(0)).squeeze()
        alpha = nn.functional.softplus(raw[0]).clamp(max=100.0) + 1e-3
        beta = nn.functional.softplus(raw[1]).clamp(max=100.0) + 1e-3
        return alpha, beta

    def predict_pair(self, mem_a: torch.Tensor, mem_b: torch.Tensor, last_features: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.predict_pair_distribution(mem_a, mem_b, last_features)
        return alpha / (alpha + beta)


def beta_nll_loss(alpha: torch.Tensor, beta: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of binary labels under Beta(alpha, beta) --
    penalizes confident-and-wrong more than honestly-uncertain-and-wrong."""
    dist = Beta(alpha, beta)
    labels_clamped = labels.clamp(1e-4, 1 - 1e-4)
    return -dist.log_prob(labels_clamped).mean()


def forward_and_score_with_uncertainty(model: TGNLite, graph: DynamicGraphData, device="cpu"):
    """Uses each pair's OWN memory snapshot (at its own last event),
    not a single shared final memory -- the actual fix."""
    num_nodes = len(graph.node_ids)
    _, pair_memory_snapshots = model.run_events(num_nodes, graph.events, device=device)

    last_features: dict[tuple[int, int], torch.Tensor] = {}
    for event in graph.events:
        last_features[(event.node_a, event.node_b)] = torch.from_numpy(event.features).to(device)

    alphas, betas, labels = [], [], []
    for pair, label in graph.pair_labels.items():
        if pair not in pair_memory_snapshots:
            continue
        mem_a, mem_b = pair_memory_snapshots[pair]
        a, b = model.predict_pair_distribution(mem_a, mem_b, last_features[pair])
        alphas.append(a)
        betas.append(b)
        labels.append(label)

    if not alphas:
        return torch.zeros(0), torch.zeros(0), labels
    return torch.stack(alphas), torch.stack(betas), labels


def forward_and_score(model: TGNLite, graph: DynamicGraphData, device="cpu"):
    """Point-estimate path, built on the same per-pair-snapshot fix --
    returns a logit-like score for compatibility with existing
    AP/precision/recall evaluation code."""
    alphas, betas, labels = forward_and_score_with_uncertainty(model, graph, device=device)
    if alphas.numel() == 0:
        return torch.zeros(0), labels
    means = alphas / (alphas + betas)
    means_clamped = means.clamp(1e-4, 1 - 1e-4)
    logits = torch.log(means_clamped / (1 - means_clamped))
    return logits, labels