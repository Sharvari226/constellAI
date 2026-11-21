param(
  [string]$Root = "outputs/ladder_het_n300_r2",
  [int]$Expect = 18,
  [int]$Trials = 300,
  [int]$Gate = 6
)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
$env:OMP_NUM_THREADS = "1"
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

Write-Host "Waiting for $Expect finished jobs under $Root"
while ($true) {
  $n = @(Get-ChildItem $Root -Recurse -Filter done.txt -ErrorAction SilentlyContinue).Count
  $alive = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "run_ladder" }).Count
  Write-Host ("{0}  done {1}/{2}  ladder processes {3}" -f (Get-Date -Format "HH:mm"), $n, $Expect, $alive)
  if ($n -ge $Expect) { break }
  if ($alive -eq 0) { throw "ladder is not running and only $n/$Expect jobs are done; relaunch run_ladder (it resumes)" }
  Start-Sleep -Seconds 300
}

python constellai/scripts/select_and_compare.py --root $Root --target 0.98 --boot 2000 | Tee-Object -FilePath outputs/select_n300.txt
Check "select_and_compare"
python constellai/scripts/make_figures.py --root $Root --sigma 25 | Out-Null
Check "make_figures"
python constellai/scripts/freeze_final.py --root $Root --trials $Trials | Tee-Object -FilePath outputs/freeze_dryrun.txt
Check "freeze dry run"

$a = Read-Host "Review the selection above. Write and commit frozen_thresholds.json? (y/n)"
if ($a -ne "y") { Write-Host "stopped before freezing"; return }
python constellai/scripts/freeze_final.py --root $Root --trials $Trials --write
Check "freeze write"
git add frozen_thresholds.json
git commit -m "Freeze thresholds chosen on stream 4 (n=300) before the stream 7 evaluation"
Check "git commit"
git push origin feat/noisy-tracks
if ($LASTEXITCODE -ne 0) { Write-Host "push failed; the commit exists locally, continuing" }

$b = Read-Host "Run stream 7 ONCE now (n=$Trials)? (y/n)"
if ($b -ne "y") { Write-Host "stopped before stream 7"; return }
python constellai/scripts/freeze_final.py --root $Root --trials $Trials --mc-gate $Gate --execute | Tee-Object -FilePath outputs/final_stream7.log
Check "stream 7 run"
python constellai/scripts/report_final.py | Tee-Object -FilePath outputs/final_stream7_report.txt
Check "report_final"
Write-Host "Done. Report: outputs/final_stream7_report.txt ; per-policy JSONs under outputs/final_stream7"
