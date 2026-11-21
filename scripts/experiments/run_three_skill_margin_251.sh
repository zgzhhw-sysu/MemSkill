#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
parent="$base/late-pairs/semantic-writer/three-writer"
mode=${1:-margin}
case $mode in
    margin)
        out="$parent/margin"
        cmd=(experiments.controller.three_skill_margin "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA") ;;
    time|both|raw-reference)
        out="$parent/direct-allocation/$mode"
        cmd=(experiments.controller.direct_skill_allocation "$base" "$mode")
        audit="$base/source-compression/activity-relation-repairs"
        until [[ -f "$parent/margin/forecast.json" ]]; do sleep 10; done
        "$MEMSKILL_PYTHON" -m "${cmd[@]}" --register-only
        stage=outcomes
        if [[ $mode == raw-reference ]]; then stage=prepared; fi
        until [[ -f "$audit/0/$stage-complete.json" ]]; do sleep 35; done
        if [[ $mode == both ]]; then
            until [[ -f "$audit/1/outcomes-complete.json" && -f "$parent/direct-allocation/time/forecast.json" ]]; do sleep 35; done
        fi ;;
    *) exit 2 ;;
esac
mkdir -p "$out"
last=-1
until [[ -f "$out/summary.json" ]]; do
    count=$(find "$parent" -maxdepth 1 -name 'conv-*.json' | wc -l)
    if [[ $count != "$last" ]]; then
        rc=0
        python3 "$guard" run --gpus 4 -- "$MEMSKILL_PYTHON" -u -m "${cmd[@]}" >>"$out/policy.log" 2>&1 || rc=$?
        if ((rc==0)); then
            last=$count
            if [[ $mode == raw-reference ]]; then exit 0; fi
        elif ((rc!=75)); then exit "$rc"; fi
    fi
    sleep 35
done
