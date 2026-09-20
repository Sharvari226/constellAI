"""Minimal diagnostic: trains TGN-lite's uncertainty head on ONE
scenario for 3 epochs, printing loss and alpha/beta ranges after every
step -- built specifically to answer 'is this working-but-slow,
numerically broken, or actually frozen', since guessing from CPU
numbers and elapsed time alone hasn't settled that question.

Run: python -m constellai.scripts.debug_uncertainty_training
"""

import time
from datetime import datetime, timedelta

import torch

from constellai.models.tgnn.dynamic_graph import build_dynamic_graph
from constellai.models.tgnn.tgn_lite import TGNLite, beta_nll_loss, forward_and_score_with_uncertainty
from constellai.scripts.compare_baselines import make_scenario, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM

print("Building ONE scenario...")
scenario = make_scenario(seed=0, id_offset=0)

print("Building dynamic graph...")
t0 = time.time()
graph = build_dynamic_graph(scenario, OBS_START, OBS_END, HORIZON_END, STEP_TGN, MARGIN_KM, THRESHOLD_KM)
print(f"  graph built in {time.time() - t0:.2f}s -- {len(graph.events)} events, {len(graph.pair_labels)} pairs")

model = TGNLite()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    t0 = time.time()
    optimizer.zero_grad()

    print(f"  epoch {epoch}: calling forward_and_score_with_uncertainty...")
    t_fwd = time.time()
    alphas, betas, labels = forward_and_score_with_uncertainty(model, graph)
    print(f"    forward pass done in {time.time() - t_fwd:.2f}s -- "
          f"alpha range [{alphas.min().item():.4f}, {alphas.max().item():.4f}], "
          f"beta range [{betas.min().item():.4f}, {betas.max().item():.4f}]")

    if alphas.numel() == 0:
        print("    NO PAIRS TO SCORE -- graph produced zero valid pairs, this is a real problem, stopping.")
        break

    t_loss = time.time()
    loss = beta_nll_loss(alphas, betas, torch.tensor(labels, dtype=torch.float))
    print(f"    loss = {loss.item():.4f} (computed in {time.time() - t_loss:.4f}s)")

    if torch.isnan(loss) or torch.isinf(loss):
        print("    LOSS IS NaN/Inf -- found the bug.")
        break

    t_bwd = time.time()
    loss.backward()
    print(f"    backward() done in {time.time() - t_bwd:.2f}s")

    t_step = time.time()
    optimizer.step()
    print(f"    optimizer.step() done in {time.time() - t_step:.2f}s")

    print(f"  epoch {epoch} TOTAL: {time.time() - t0:.2f}s")

print("Done.")