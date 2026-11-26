#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export MEMSKILL_API_TIMEOUT_S=600 MEMSKILL_API_BASE=http://127.0.0.1:18773/v1
export MEMSKILL_BACKEND_ID=251-exchange-llama3-bf16-gpu56-duty28-qwen3-b2-duty30
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
src="$base/source-compression/writer-format-credit"
study="$work/study-linked-eight-gpu"
phase=${1:?writer-format phase}
mkdir -p "$src"
export MEMSKILL_USAGE_LOG="$src/usage.jsonl"
cmd=("$MEMSKILL_PYTHON" -u -m experiments.controller.writer_format_trial "$base" "$study" "$LOCOMO_DATA")
if [[ ! -f "$src/registration.json" ]]; then "${cmd[@]}" register; fi
if [[ $phase == shared-prepared || $phase == shared-outcomes ]]; then
    part=${phase#shared-}
    until [[ -f "$src/verified-complete.json" ]]; do sleep 15; done
    shared=("$MEMSKILL_PYTHON" -u -m experiments.controller.three_skill_shared_credit "$base" "$study" "$LOCOMO_DATA")
    "${shared[@]}" register
    export MEMSKILL_USAGE_LOG="$src/shared/usage.jsonl"
    if [[ $part == outcomes ]]; then until [[ -f "$src/shared/prepared-complete.json" ]]; do sleep 15; done; fi
    until [[ -f "$src/shared/$part-complete.json" ]]; do
        rc=0
        if [[ $part == prepared ]]; then
            python3 "$guard" run --gpus 3 -- "${shared[@]}" "$part" >>"$src/shared/$part.log" 2>&1 || rc=$?
        elif bash scripts/experiments/run_witness_four_paths_251.sh check-writer; then
            "${shared[@]}" "$part" >>"$src/shared/$part.log" 2>&1 || rc=$?
        fi
        if ((rc!=0 && rc!=75)); then exit "$rc"; fi
        [[ -f "$src/shared/$part-complete.json" ]] || sleep 20
    done
    exit 0
fi
if [[ $phase == sampling ]]; then
    until [[ -f "$src/prepared-complete.json" && -f "$src/verified-complete.json" ]]; do sleep 15; done
    survey=("$MEMSKILL_PYTHON" -u -m experiments.controller.writer_format_sampling "$base")
    "${survey[@]}" register
    while [[ ! -f "$src/sampling/sample2.json" ]]; do
        if bash scripts/experiments/run_witness_four_paths_251.sh check-writer; then
            "${survey[@]}" read >>"$src/sampling/reader.log" 2>&1
        else
            sleep 15
        fi
    done
    until [[ -f "$src/readers-complete.json" ]]; do sleep 35; done
    "${survey[@]}" compare
    exit 0
fi
if [[ $phase == identification ]]; then
    prev=
    while [[ ! -f "$src/identification/summary.json" ]]; do
        if [[ -f "$base/late-pairs/semantic-writer/three-writer/writer-format-history/forecast.json" ]]; then
            state=$(find "$src/prepared" "$src/questions" -name '*.json' -type f -printf '%p\n' 2>/dev/null | sort || true)
            if [[ $state != "$prev" || -f "$src/readers-complete.json" ]]; then
                rc=0
                python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.writer_format_audit "$base" >>"$src/identification.log" 2>&1 || rc=$?
                if ((rc==0)); then prev=$state; elif ((rc!=75)); then exit "$rc"; fi
            fi
        fi
        [[ -f "$src/identification/summary.json" ]] || sleep 30
    done
    exit 0
fi
case $phase in
    prepared) ;;
    readers) until [[ -f "$src/verified-complete.json" ]]; do sleep 20; done ;;
    verified) ;;
    *) exit 2 ;;
esac
mapfile -t names < <(python3 - "$src/registration.json" "$phase" <<'PYTHON'
import json, sys
flags = json.load(open(sys.argv[1]))["flags"]
print("\n".join(k for k in flags if sys.argv[2] != "verified" or flags[k]))
PYTHON
)
until [[ -f "$src/$phase-complete.json" ]]; do
    rc=0
    if [[ $phase == prepared ]]; then
        python3 "$guard" run --gpus 3 -- "${cmd[@]}" "$phase" >>"$src/$phase.log" 2>&1 || rc=$?
    else
        ready=false; finished=0
        for name in "${names[@]}"; do
            if [[ -f "$src/$phase/$name.json" ]]; then finished=$((finished+1))
            elif [[ -f "$src/prepared/$name.json" ]]; then ready=true
            fi
        done
        if ((finished==${#names[@]})) || { [[ $ready == true ]] && bash scripts/experiments/run_witness_four_paths_251.sh check-writer; }; then
            "${cmd[@]}" "$phase" >>"$src/$phase.log" 2>&1 || rc=$?
        fi
    fi
    if ((rc!=0 && rc!=75)); then exit "$rc"; fi
    [[ -f "$src/$phase-complete.json" ]] || sleep 12
done
