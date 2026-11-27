"""Fetch a reproducible CelesTrak Starlink TLE snapshot and run existing M1/M2 code."""
from __future__ import annotations
import argparse, csv, json, sys, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
REPO = DEMO_ROOT.parent
DATA = DEMO_ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--step-minutes", type=float, default=2.0)
    p.add_argument("--margin-km", type=float, default=50.0)
    p.add_argument("--distance-km", type=float, default=300.0)
    p.add_argument("--min-closing-rate", type=float, default=0.0)
    p.add_argument("--offline", action="store_true", help="Reuse existing raw TLE snapshot")
    a = p.parse_args()
    if a.count < 2 or a.step_minutes <= 0 or a.hours <= 0:
        p.error("count must be >=2 and hours/step-minutes must be positive")
    sys.path.insert(0, str(REPO))
    from constellai.orbital_mechanics.tle import parse_tle_file
    from constellai.orbital_mechanics.propagation import propagate
    from constellai.orbital_mechanics.regime import altitude_band
    from constellai.graph.filters import candidate_pairs_by_regime
    from constellai.graph.build import build_graph

    RAW.mkdir(parents=True, exist_ok=True); PROCESSED.mkdir(parents=True, exist_ok=True)
    tle_path = RAW / "starlink_snapshot.tle"
    if not a.offline:
        req = urllib.request.Request(URL, headers={"User-Agent": "ConstellAI-capstone-demo/1.0"})
        with urllib.request.urlopen(req, timeout=45) as response:
            payload = response.read().decode("utf-8")
        tle_path.write_text(payload, encoding="utf-8")
    if not tle_path.exists(): raise FileNotFoundError("No saved TLE. Run without --offline once.")
    records = parse_tle_file(tle_path)[:a.count]
    if len(records) < 2: raise RuntimeError(f"Only {len(records)} valid TLEs found")
    now = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
    start, end = now, now + timedelta(hours=a.hours)
    rows=[]
    for r in records:
        band=altitude_band(r); state=propagate(r, start)
        rows.append({"satellite_id":r.satellite_id,"name":r.name,"perigee_alt_km":band.perigee_alt_km,"apogee_alt_km":band.apogee_alt_km,"inclination_deg":r.satrec.inclo*180/3.141592653589793,"raan_deg":r.satrec.nodeo*180/3.141592653589793,"mean_motion_rev_day":r.satrec.no_kozai*1440/(2*3.141592653589793),"x_km":float(state.position_km[0]),"y_km":float(state.position_km[1]),"z_km":float(state.position_km[2]),"vx_km_s":float(state.velocity_km_s[0]),"vy_km_s":float(state.velocity_km_s[1]),"vz_km_s":float(state.velocity_km_s[2])})
    with (PROCESSED/"satellites.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    candidates=candidate_pairs_by_regime(records, margin_km=a.margin_km)
    edges=build_graph(records,start,end,timedelta(minutes=a.step_minutes),a.margin_km,a.distance_km,a.min_closing_rate)
    with (PROCESSED/"candidate_pairs.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["satellite_a","satellite_b"]);w.writerows([(x.satellite_id,y.satellite_id) for x,y in candidates])
    edge_rows=[]
    for edge in edges:
        e=edge.event
        edge_rows.append({"satellite_a":e.satellite_id_a,"satellite_b":e.satellite_id_b,"tca":e.tca.isoformat() if hasattr(e.tca,"isoformat") else str(e.tca),"miss_distance_km":e.miss_distance_km,"relative_speed_km_s":e.relative_speed_km_s})
    with (PROCESSED/"screened_edges.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["satellite_a","satellite_b","tca","miss_distance_km","relative_speed_km_s"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(edge_rows)
    meta={"source":URL,"retrieved_at_utc":datetime.now(timezone.utc).isoformat(),"snapshot_epoch_utc":start.isoformat()+"Z","satellites_requested":a.count,"satellites_parsed":len(records),"exhaustive_pairs":len(records)*(len(records)-1)//2,"coarse_candidates":len(candidates),"screened_edges":len(edges),"screening_window_hours":a.hours,"step_minutes":a.step_minutes,"margin_km":a.margin_km,"distance_threshold_km":a.distance_km,"min_closing_rate_km_s":a.min_closing_rate,"frame":"TEME","note":"Positions are TEME states at one common epoch. Pair screening uses the repository's existing conjunction screen_pair implementation."}
    (PROCESSED/"snapshot_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps(meta,indent=2))

if __name__ == "__main__": main()
