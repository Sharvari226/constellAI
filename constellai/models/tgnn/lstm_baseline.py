"""Step-2 baseline on the M3 ladder: plain per-pair LSTM, no graph
structure. Must be beaten by the eventual TGNN (Step 3) -- if it isn't,
the graph structure isn't earning its complexity."""

from __future__ import annotations

import torch
from torch import nn


class PairRiskLSTM(nn.Module):
    """Consumes a (T, 4) relative-dynamics window per pair, outputs a
    single risk probability. Deliberately simple: this is the floor
    the TGNN must clear, not a model worth tuning heavily."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, input_dim) -> (batch,) risk logits."""
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def train_one_epoch(model, examples, optimizer, device="cpu"):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for ex in examples:
        x = torch.from_numpy(ex.features).unsqueeze(0).to(device)  # (1, T, 4)
        y = torch.tensor([float(ex.label)], device=device)

        optimizer.zero_grad()
        logit = model(x)
        loss = loss_fn(logit, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(examples)