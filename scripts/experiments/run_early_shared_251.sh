#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MEMSKILL_BACKEND_ID=251-exchange-llama3-bf16-gpu56-duty28-qwen3-b2-duty30
export MEMSKILL_API_BASE=http://127.0.0.1:18773/v1 MEMSKILL_API_TIMEOUT_S=600
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
base="$work/runs/paired-251-linked-eight"
early="$base/source-compression/earliest-three-skills"
out="$early/shared-weak-credit"
mkdir -p "$out"
export MEMSKILL_USAGE_LOG="$out/usage.jsonl"
cmd=("$MEMSKILL_PYTHON" -u -m experiments.controller.early_shared_credit "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA")
"${cmd[@]}" register
until [[ -f "$out/final.json" ]]; do
    parent="$early/judged"
    for phase in audit additional certified prepared outcomes; do
        ready=false
        for file in "$parent"/conv-*.json; do
            if [[ -f $file && ! -f "$out/$phase/${file##*/}" ]]; then ready=true; break; fi
        done
        source_dir=$parent
        parent="$out/$phase"
        if [[ $ready != true ]]; then continue; fi
        rc=0
        case $phase in
            additional|prepared)
                if python3 - "$phase" "$source_dir" "$parent" <<'PYTHON'
import json, sys
from pathlib import Path
stage, src, dest = sys.argv[1:]
for f in Path(src).glob("conv-*.json"):
    if (Path(dest) / f.name).exists():
        continue
    for c in json.loads(f.read_bytes()).get("candidates", {}).values():
        if stage == "prepared" and c["status"] == "certified":
            sys.exit(0)
        if stage == "additional" and not any("veto" in a for a in c["audits"].values()) and not all(c["audits"].values()):
            sys.exit(0)
sys.exit(1)
PYTHON
                then
                    python3 "$guard" run --gpus 3 -- "${cmd[@]}" "$phase" >>"$out/$phase.log" 2>&1 || rc=$?
                else
                    "${cmd[@]}" "$phase" >>"$out/$phase.log" 2>&1 || rc=$?
                fi ;;
            certified|outcomes) if bash scripts/experiments/run_witness_four_paths_251.sh check-writer; then "${cmd[@]}" "$phase" >>"$out/$phase.log" 2>&1 || rc=$?; fi ;;
            audit) "${cmd[@]}" "$phase" >>"$out/$phase.log" 2>&1 || rc=$? ;;
        esac
        if ((rc!=0 && rc!=75)); then exit "$rc"; fi
    done
    if [[ ! -f "$out/initial.json" || -f "$early/judged-complete.json" ]]; then "${cmd[@]}" markers; fi
    [[ -f "$out/final.json" ]] || sleep 25
done
