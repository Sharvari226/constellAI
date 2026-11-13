"""Freeze thresholds from stream 4 and run the one-time stream 7 evaluation.

Default: dry run (prints the selection and commands). --write saves frozen_thresholds.json.
--execute runs stream 7 and requires the frozen file to be committed and clean. One shot only.
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import select_and_compare as sc  # noqa: E402

FAMS = ["A0", "E", "E_hcw", "Ehcw_k", "F600", "F400", "F200", "B"]
FROZEN = ROOT / "frozen_thresholds.json"
DONE = ROOT / "outputs" / "final_stream7" / "FINAL_DONE.txt"


def flags(name):
    m = re.fullmatch(r"dist_D(.+)", name)
    if m:
        return ["--distance-trigger", m.group(1)]

    m = re.fullmatch(r"heuristic_tau(.+)", name)
    if m:
        return ["--heuristic-tau", m.group(1)]

    m = re.fullmatch(r"hcw_tau(.+)", name)
    if m:
        return ["--hcw-tau", m.group(1)]

    m = re.fullmatch(r"hcwm_k(.+)", name)
    if m:
        return ["--hcw-margin-k", m.group(1)]

    m = re.fullmatch(r"prob_(analytic|oracle)_p(.+)_h(\d+)", name)
    if m:
        return [
            "--prob-trigger", m.group(2),
            "--prob-horizon", m.group(3),
            "--forecast-kind", m.group(1)
        ]

    raise ValueError("unrecognised policy name: " + name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--sigma", type=float, default=25.0)
    ap.add_argument("--trials", type=int, required=True)
    ap.add_argument("--sigma-choices", type=float, nargs="*", default=[5.0, 50.0])
    ap.add_argument("--mc-gate", type=float, default=0.0)
    ap.add_argument("--noise-seed", type=int, default=12345)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--expect-jobs", type=int, default=18)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()

    n_done = len(list(Path(a.root).glob("*/done.txt")))
    if n_done < a.expect_jobs and not a.allow_partial:
        sys.exit(f"INCOMPLETE: {n_done}/{a.expect_jobs} jobs done under {a.root}; selection would be misleading")
    data = sc.load(a.root)
    chosen = {}

    for fam in FAMS:
        name = sc.choose(data, a.sigma, fam, a.target)

        if name is None:
            print(f"{fam:<7} no threshold reaches {a.target} on stream 4 -> excluded")
            continue

        free, dv = data[(a.sigma, sc.VAL, name)]

        chosen[fam] = {
            "policy": name,
            "val_free": float(free.mean()),
            "val_dv": float(dv.mean()),
            "val_n": int(len(free))
        }

        print(
            f"{fam:<7} {name:<26} "
            f"val free {free.mean():.3f}  "
            f"val dv {dv.mean():.3f}  n={len(free)}"
        )

    if not a.write and not a.execute:
        print("dry run; use --write to save", FROZEN.name)

    if a.write:
        if FROZEN.exists():
            sys.exit("frozen_thresholds.json exists; refusing to overwrite")

        FROZEN.write_text(
            json.dumps({
                "source_root": a.root,
                "target": a.target,
                "sigma": a.sigma,
                "chosen": chosen
            }, indent=2),
            encoding="utf-8"
        )

        print("wrote", FROZEN)

    if not a.execute:
        return

    if DONE.exists():
        sys.exit("stream 7 already run; refusing a second run")

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", FROZEN.name],
        cwd=ROOT,
        capture_output=True
    ).returncode == 0

    clean = subprocess.run(
        ["git", "status", "--porcelain", "--", FROZEN.name],
        cwd=ROOT,
        capture_output=True,
        text=True
    ).stdout.strip() == ""

    if not (tracked and clean):
        sys.exit("commit frozen_thresholds.json first (tracked and clean required)")

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["chosen"]

    outdir = ROOT / "outputs" / "final_stream7"
    outdir.mkdir(parents=True, exist_ok=True)

    for fam, info in frozen.items():
        name = info["policy"]
        d = outdir / fam
        d.mkdir(exist_ok=True)

        cmd = [
            sys.executable,
            "constellai/scripts/evaluate_tracked.py",
            "--trials", str(a.trials),
            "--preset", "short",
            "--sigma-p", str(a.sigma),
            "--noise-seed", str(a.noise_seed),
            "--final",
            "--out-dir", str(d)
        ] + flags(name)

        if a.sigma_choices:
            cmd += ["--sigma-choices"] + [str(s) for s in a.sigma_choices]

        if a.mc_gate:
            cmd += ["--mc-gate", str(a.mc_gate)]

        print("RUN", " ".join(cmd), flush=True)

        rc = subprocess.run(cmd, cwd=ROOT).returncode

        if rc != 0 or not (d / f"eval_{name}_stream7.json").exists():
            sys.exit(
                f"{fam}: failed or unexpected output name "
                f"(expected eval_{name}_stream7.json); stopping"
            )

    DONE.write_text("ok", encoding="utf-8")
    print("stream 7 complete")


if __name__ == "__main__":
    main()
