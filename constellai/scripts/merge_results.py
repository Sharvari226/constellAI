# Merge Monte Carlo result files written by parallel run_monte_carlo processes.
#
# Run: python -m constellai.scripts.merge_results outputs/m6/mc_*.json --out outputs/m6/monte_carlo.json
# Cell keys include tier, simulator, filter, policy, trial count and seed, so cells
# from different processes never collide; an identical key in two files keeps the later file's cell.

from __future__ import annotations

import argparse

from constellai.evaluation.monte_carlo import load_results, save_results


def merge_results(results_list: list) -> dict:
    merged = {"meta": {}, "cells": {}}
    for index, results in enumerate(results_list):
        if index == 0:
            merged["meta"] = dict(results.get("meta", {}))
        merged["cells"].update(results["cells"])
    merged["meta"]["merged_files"] = len(results_list)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge Monte Carlo result files")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merged = merge_results([load_results(path) for path in args.inputs])
    save_results(args.out, merged)
    print(f"merged {len(args.inputs)} files, {len(merged['cells'])} cells -> {args.out}")


if __name__ == "__main__":
    main()