from pathlib import Path
import json
import pandas as pd

DEMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DEMO_ROOT.parent
PROCESSED = REPO_ROOT / "data" / "processed"
FINAL_CSV = REPO_ROOT / "artifacts" / "04_final_results" / "final_stream7_report.csv"

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def load_snapshot():
    meta=read_json(PROCESSED/"snapshot_metadata.json")
    sat=pd.read_csv(PROCESSED/"satellites.csv") if (PROCESSED/"satellites.csv").exists() else pd.DataFrame()
    candidates=pd.read_csv(PROCESSED/"candidate_pairs.csv") if (PROCESSED/"candidate_pairs.csv").exists() else pd.DataFrame(columns=["satellite_a","satellite_b"])
    edges=pd.read_csv(PROCESSED/"screened_edges.csv") if (PROCESSED/"screened_edges.csv").exists() else pd.DataFrame(columns=["satellite_a","satellite_b","tca","miss_distance_km","relative_speed_km_s"])
    return meta,sat,candidates,edges

def load_final_results():
    return pd.read_csv(FINAL_CSV) if FINAL_CSV.exists() else pd.DataFrame()
