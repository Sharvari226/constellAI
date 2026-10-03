# ConstellAI visualization

This folder presents **CURRENT IMPLEMENTATION / CURRENT RESULTS** from stable baseline `20dad4d` only. It does not alter experiment configuration or training, graph construction, model, evaluation, uncertainty, MARL, or safety-filter code. The generated M2 figures call the canonical graph builder and existing validation gate with the existing false-negative-gate scenario and settings.

## Figures available

| Figure | Source | Reproduce |
|---|---|---|
| `figures/m2_graph_stages.svg` | Current coarse altitude candidate function and fine screening function, run on `build_scenario(seed=0)` from `constellai.scripts.run_false_negative_gate`; exhaustive stage is all unordered pairs in that same actual generated scenario. | `python -m visualization.generate_figures` |
| `figures/m2_sparsification.svg` | Counts from the same scenario and current pipeline: exhaustive pairs, altitude-regime candidates, and fine-screen edges. | `python -m visualization.generate_figures` |
| `figures/m2_false_negative_validation.svg` | `constellai.graph.validation.run_false_negative_gate` on the same scenario. Shows baseline relevant pairs, retained pairs, missed pairs, and the existing overall false-negative rate. The current gate does not expose separate coarse/fine false-negative counts. | `python -m visualization.generate_figures` |
| `figures/constellai_architecture.svg` | Explanatory system flow based on the repository architecture description. Not an experimental result. | `python -m visualization.generate_figures` |

The graph layouts use deterministic circular node positions to make topology legible. Those positions do **not** depict orbital coordinates. The graph-stage and false-negative figures use the gate script’s 18-satellite, single-shell synthetic scenario (500 km ± 3 km), two-hour window at two-minute steps, 300 km screening distance, and 50 km regime margin. This demonstrates what the current code produces for that scenario; it is not a general scale or real-catalog claim.

## Existing results not plotted

- The stable baseline contains no committed M3 result files or saved predictions. `compare_baselines.py` prints metrics when run, but those outputs are not preserved as current results, so no M3 metric or prediction figures are included.
- The baseline comparison does not report ECE or preserve per-bin calibration data. No reliability diagram or ECE figure is included.
- Epoch-wise losses are printed in some script runs, but no saved epoch history is available. No loss curve is included.
- M4 episode return, fuel/delta-v, and relative-trajectory output are not saved by the current experiments. No M4 result plots are included.
- M5 proposed and filtered actions or intervention statistics are not saved by a current experiment. No M5 result plots are included.

The comparison script explicitly documents timestep and test-set differences for TGN-lite. No metrics have been rerun for this visualization branch. Treat these figures as a presentation of current code and available M2 results, not as final validated results.

## CURRENT IMPLEMENTATION / CURRENT RESULTS vs FUTURE METHODOLOGICAL FIXES

This branch visualizes the stable implementation at `20dad4d`. Known methodological inconsistencies—including M3 timestep and test-set differences and synthetic scenarios—remain intentionally unresolved. Any future standardization, data changes, reruns, or methodological corrections belong in a separate phase and branch.

## Runtime

The repository declares Matplotlib and NumPy in `pyproject.toml`; no dependency or version changes were made. The plotting script itself uses only the Python standard library and writes SVG, so it can run in environments where the declared plotting dependency is not installed. Run from the repository root:

```powershell
python -m visualization.generate_figures
```

The script does not import model training routines, start training, or set or mutate random seeds. M2 scenario creation uses the existing seeded gate scenario builder. Output files are written to `visualization/figures/`.
