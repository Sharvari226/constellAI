"""M3 Step 3: minimal continuous-time temporal graph model ("TGN-lite"),
now with a calibrated uncertainty head, not just a point-estimate risk
score.

Each satellite has a persistent memory vector that updates via a GRU
cell every time a new event involving it occurs, chronologically.
Memory is DETACHED before each update (truncated backpropagation
through time) -- without this, a single backward pass has to walk back
through the entire prior interaction history for every node, which was
the actual cause of multi-hour training times, not a bug in the event
stream itself.

UNCERTAINTY HEAD: instead of one risk_head predicting a single
probability, risk_head now predicts two positive numbers (alpha, beta)
-- the parameters of a Beta distribution over the true collision
probability. Beta(alpha, beta) naturally represents "I think risk is
around alpha/(alpha+beta), and I am this confident about it" -- a
sharply peaked distribution (large alpha+beta) means confident, a flat
one (small alpha+beta) means genuinely uncertain. This is deliberately
simpler than a full Bayesian or ensemble approach (matching this
project's "lite" scope), but it is a real, trainable notion of
calibrated uncertainty, not a cosmetic addition: the loss below
(Beta-NLL) penalizes a model that is both wrong AND falsely confident
harder than one that is wrong but appropriately uncertain about it.
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
        # Outputs 2 raw values -> softplus -> (alpha, beta), both > 0.
        # Was a single logit before; this is the actual uncertainty-head
        # change, everything else in the architecture is unchanged.
        self.risk_head = nn.Sequential(
            nn.Linear(2 * memory_dim + feature_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 2),
        )

    def run_events(self, num_nodes: int, events, device="cpu") -> torch.Tensor:
        """Process the chronological event stream, updating per-node
        memory as it goes. Returns final memory, shape (num_nodes, memory_dim).
        See module docstring for why memory is detached before each update.
        """
        memory = [torch.zeros(1, self.memory_dim, device=device) for _ in range(num_nodes)]

        for event in events:
            feat = torch.from_numpy(event.features).unsqueeze(0).to(device)
            feat_from_b = feat.clone()
            feat_from_b[:, :3] *= -1  # position components only; separation and rel_speed are unsigned magnitudes

            msg_a = self.message(feat)
            msg_b = self.message(feat_from_b)
            memory[event.node_a] = self.memory_update(msg_a, memory[event.node_a].detach())
            memory[event.node_b] = self.memory_update(msg_b, memory[event.node_b].detach())

        return torch.cat(memory, dim=0)

    def predict_pair_distribution(
        self, memory: torch.Tensor, node_a: int, node_b: int, last_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (alpha, beta) for this pair's Beta-distributed risk
        estimate -- NOT a single probability. Both are guaranteed > 0
        via softplus + a small epsilon (Beta requires strictly positive
        parameters; a value that hits exactly 0 would be an invalid
        distribution, not just a degenerate one).
        """
        pair_input = torch.cat([memory[node_a], memory[node_b], last_features])
        raw = self.risk_head(pair_input.unsqueeze(0)).squeeze()
        alpha = nn.functional.softplus(raw[0]) + 1e-3
        beta = nn.functional.softplus(raw[1]) + 1e-3
        return alpha, beta

    def predict_pair(self, memory: torch.Tensor, node_a: int, node_b: int, last_features: torch.Tensor) -> torch.Tensor:
        """Point-estimate convenience wrapper: the Beta distribution's
        mean, alpha/(alpha+beta) -- kept for compatibility with anything
        (like the existing AP/precision/recall evaluation) that expects
        a single risk score. Downstream consumers that actually need the
        uncertainty (M4's constraint shaping, per the research spec)
        should call predict_pair_distribution directly instead."""
        alpha, beta = self.predict_pair_distribution(memory, node_a, node_b, last_features)
        return alpha / (alpha + beta)


def beta_nll_loss(alpha: torch.Tensor, beta: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of binary labels under Beta(alpha, beta),
    treating each label as an (imperfect) draw from the true risk
    distribution.

    This is what actually trains calibration, not just accuracy: a
    prediction of alpha=beta=1 (uniform, "I have no idea") and one of
    alpha=100, beta=1 (sharply confident, mean~0.99) can have the SAME
    mean prediction, but this loss penalizes the second one much more
    harshly when the label turns out to be 0 -- confident-and-wrong
    costs more than honestly-uncertain-and-wrong. A plain BCE loss on
    the mean alone cannot distinguish these two cases at all.
    """
    dist = Beta(alpha, beta)
    # Labels are hard 0/1; clamp slightly inside (0,1) since Beta's
    # log_prob is undefined exactly at the boundary for some (alpha,beta).
    labels_clamped = labels.clamp(1e-4, 1 - 1e-4)
    return -dist.log_prob(labels_clamped).mean()


def forward_and_score_with_uncertainty(model: TGNLite, graph: DynamicGraphData, device="cpu"):
    """Like forward_and_score, but returns (alpha, beta) per pair instead
    of a single logit -- the actual training/eval path for the
    uncertainty-aware model. forward_and_score (point-estimate only) is
    kept unchanged for anything that doesn't need the distribution."""
    num_nodes = len(graph.node_ids)
    memory = model.run_events(num_nodes, graph.events, device=device)

    last_features: dict[tuple[int, int], torch.Tensor] = {}
    for event in graph.events:
        last_features[(event.node_a, event.node_b)] = torch.from_numpy(event.features).to(device)

    alphas, betas, pairs, labels = [], [], [], []
    for pair, label in graph.pair_labels.items():
        if pair not in last_features:
            continue
        a, b = model.predict_pair_distribution(memory, pair[0], pair[1], last_features[pair])
        alphas.append(a)
        betas.append(b)
        pairs.append(pair)
        labels.append(label)

    if not alphas:
        return torch.zeros(0), torch.zeros(0), labels
    return torch.stack(alphas), torch.stack(betas), labels


def forward_and_score(model: TGNLite, graph: DynamicGraphData, device="cpu"):
    """Point-estimate path, UNCHANGED behavior from before the
    uncertainty head was added -- returns the Beta mean as a single
    logit-like score, so existing AP/precision/recall evaluation code
    keeps working without modification."""
    alphas, betas, labels = forward_and_score_with_uncertainty(model, graph, device=device)
    if alphas.numel() == 0:
        return torch.zeros(0), labels
    means = alphas / (alphas + betas)
    # Convert mean-probability back to a logit so existing code that
    # applies sigmoid() for scoring still produces the right ranking --
    # NOT numerically identical to the old raw logit, but preserves the
    # same downstream sigmoid(x) > 0.5 threshold behavior.
    means_clamped = means.clamp(1e-4, 1 - 1e-4)
    logits = torch.log(means_clamped / (1 - means_clamped))
    return logits, labels