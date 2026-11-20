#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
base="$work/runs/paired-251-linked-eight"
src="$base/source-compression"
root="$src/matched-span-rank"
args=("$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA")
phase=${1:?expected encode, original, expanded, calibrated-* or consensus-*}
module=experiments.controller.matched_span_policy
policy_root="$root"
case $phase in
    calibrated-*|consensus-*)
        family=${phase%%-*}
        phase=${phase#*-}
        [[ $phase == original || $phase == expanded ]] || exit 2
        policy_root="$root/$family"
        if [[ $family == calibrated ]]; then
            module=experiments.controller.matched_span_calibration
            parent_root="$root"
            marker=edge-audit.json
        else
            module=experiments.controller.matched_span_consensus
            parent_root="$root/calibrated"
            marker=votes.json
        fi
        until [[ -f "$parent_root/$phase/edge-audit.json" ]]; do sleep 30; done
        if [[ $phase == expanded ]]; then
            until [[ -f "$policy_root/original/$marker" ]]; do sleep 30; done
        fi ;;
esac
mkdir -p "$root"
case $phase in
    encode)
        while [[ ! -f "$root/earlier-sources/complete.json" ]]; do
            rc=0
            python3 "$guard" run --gpus 3 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.matched_span_data "${args[@]}" >>"$root/encode.log" 2>&1 || rc=$?
            if (( rc==75 )); then sleep 30; elif (( rc!=0 )); then exit "$rc"; fi
        done
        exit ;;
    original) ;;
    expanded)
        until [[ -f "$root/earlier-sources/complete.json" && -f "$src/legacy-terminal-symmetry/summary.json" && -f "$root/original/edge-audit.json" ]]; do sleep 30; done ;;
    *) exit 2 ;;
esac
last=''
result="$policy_root/$phase"
mkdir -p "$result"
until [[ -f "$result/decision.json" && -f "$result/donor/summary.json" ]]; do
    state=$(python3 - "$base" <<'PY'
from pathlib import Path
import sys
base=Path(sys.argv[1])
patterns=['late-pairs/outcomes/*/pair.json','late-pairs/donor-factorial/*/complete.json','late-pairs/cross-window-prefix/*/pair.json','source-compression/witness-passages/four-paths/cells/*/complete.json']
print([len(list(base.glob(p))) for p in patterns])
PY
)
    if [[ $state != "$last" ]]; then
        rc=0
        python3 "$guard" run --gpus 3 -- "$MEMSKILL_PYTHON" -u -m "$module" "${args[@]}" "$phase" >>"$result/policy.log" 2>&1 || rc=$?
        if (( rc==0 )); then last=$state; elif (( rc!=75 )); then exit "$rc"; fi
    fi
    sleep 35
done
