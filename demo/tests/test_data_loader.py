from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dashboard.data_loader import load_snapshot

def test_missing_snapshot_is_empty_not_fabricated():
    meta,sat,candidates,edges=load_snapshot()
    assert meta is None or isinstance(meta,dict)
    assert sat.empty or {"satellite_id","x_km","y_km","z_km"}.issubset(sat.columns)
    assert {"satellite_a","satellite_b"}.issubset(candidates.columns)
    assert {"satellite_a","satellite_b","miss_distance_km"}.issubset(edges.columns)
