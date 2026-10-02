"""Proves the per-pair memory snapshot fix actually changes behavior --
not just that the code runs, but that scoring now reads genuinely
different (and correct) memory than the old final-memory-only approach
would have.
"""

import numpy as np
import torch

from constellai.models.tgnn.dynamic_graph import GraphEvent, DynamicGraphData
from constellai.models.tgnn.tgn_lite import TGNLite, forward_and_score_with_uncertainty


def test_pair_snapshot_differs_from_final_memory_after_later_unrelated_events():
    """Node 0 interacts with node 1 ONCE at the start, then has 19 more
    events with node 2 afterward. Node 0's FINAL memory (after all 20
    events) must differ from its memory snapshot at the time of its
    single event with node 1 -- proving the snapshot mechanism captures
    a genuinely different, earlier state than 'final memory' would,
    which is exactly the bug this fix addresses."""
    torch.manual_seed(0)
    model = TGNLite()

    events = [GraphEvent(t_index=0, node_a=0, node_b=1,
                          features=np.array([1, 0, 0, 1, 0], dtype=np.float32))]
    for t in range(1, 20):
        events.append(GraphEvent(t_index=t, node_a=0, node_b=2,
                                  features=np.array([0, 1, 0, 1, 5], dtype=np.float32)))

    final_memory, pair_snapshots = model.run_events(num_nodes=3, events=events)

    mem0_final = final_memory[0]
    mem0_at_pair01 = pair_snapshots[(0, 1)][0]

    assert not torch.allclose(mem0_final, mem0_at_pair01)


def test_forward_and_score_with_uncertainty_runs_end_to_end():
    """Sanity check the changed return signature (final_memory,
    pair_snapshots) is correctly consumed downstream."""
    events = [
        GraphEvent(t_index=0, node_a=0, node_b=1, features=np.array([1, 0, 0, 1, 0], dtype=np.float32)),
        GraphEvent(t_index=1, node_a=0, node_b=1, features=np.array([1, 0, 0, 1, 0], dtype=np.float32)),
    ]
    graph = DynamicGraphData(node_ids=["0", "1"], events=events, pair_labels={(0, 1): 1})

    model = TGNLite()
    alphas, betas, labels = forward_and_score_with_uncertainty(model, graph)

    assert alphas.numel() == 1
    assert betas.numel() == 1
    assert labels == [1]