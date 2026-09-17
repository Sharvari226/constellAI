"""Smoke-train the M3 Step-2 LSTM baseline on synthetic satellites.

Run: python scripts/train_lstm_baseline.py
"""

from datetime import datetime, timedelta

import torch

from constellai.models.dataset import build_pair_examples
from constellai.models.lstm_baseline import PairRiskLSTM, train_one_epoch
from constellai.orbital_mechanics.synthetic import make_circular_satellite

# Mixed altitudes on purpose: a tight low-altitude cluster (some pairs
# risky) plus a well-separated high shell (no pairs risky) -- so the
# label set isn't all-1s or all-0s.
LOW_SHELL = [
    make_circular_satellite(satellite_id=100 + i, altitude_km=500.0 + i * 3)
    for i in range(6)
]
HIGH_SHELL = [
    make_circular_satellite(satellite_id=200 + i, altitude_km=1500.0 + i * 3)
    for i in range(6)
]
RECORDS = LOW_SHELL + HIGH_SHELL

START = datetime(2026, 1, 1)
END = START + timedelta(hours=1)
STEP = timedelta(minutes=5)
THRESHOLD_KM = 50.0


def main():
    examples = build_pair_examples(RECORDS, START, END, STEP, THRESHOLD_KM)
    labels = [e.label for e in examples]
    print(f"examples: {len(examples)}, positive: {sum(labels)}/{len(labels)}")

    model = PairRiskLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        loss = train_one_epoch(model, examples, optimizer)
        print(f"epoch {epoch}: loss={loss:.4f}")


if __name__ == "__main__":
    main()