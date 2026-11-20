#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
base="$work/runs/paired-251-linked-eight"
out="$base/source-compression/query-displacement"
args=("$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA")
mkdir -p "$out"
case ${1:?expected encode, policy or mass} in
    encode)
        until [[ -f "$out/encoded.json" ]]; do
            rc=0
            python3 "$guard" run --gpus 3 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.query_displacement_data "${args[@]}" --limit 12 >>"$out/encode.log" 2>&1 || rc=$?
            if (( rc==75 )); then sleep 30; elif (( rc!=0 )); then exit "$rc"; fi
        done ;;
    policy|mass)
        result="$out"
        metric=rank
        if [[ $1 == mass ]]; then result="$out/support-mass"; metric=mass; fi
        mkdir -p "$result"
        last=''
        until [[ -f "$result/full/decision.json" && -f "$result/full/long-prefix/summary.json" && -f "$result/full/donor/summary.json" ]]; do
            if [[ -f "$out/encoded.json" && ( $metric == rank || -f "$out/full/long-prefix/forecast.json" ) ]]; then
                state=$(python3 - "$out" <<'PY'
from pathlib import Path
import sys
base=Path(sys.argv[1]).parents[1]
items=[('late-pairs/outcomes','*/pair.json'),('late-pairs/donor-factorial','*/complete.json'),('late-pairs/cross-window-prefix','*/pair.json'),('source-compression/witness-passages/four-paths/cells','*/complete.json')]
print([len(list((base/folder).glob(pattern))) for folder,pattern in items])
PY
)
                if [[ $state != "$last" ]]; then
                    rc=0
                    python3 "$guard" run --gpus 3 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.query_displacement_policy "${args[@]}" --metric "$metric" >>"$result/policy.log" 2>&1 || rc=$?
                    if (( rc==0 )); then last=$state; elif (( rc!=75 )); then exit "$rc"; fi
                fi
            fi
            sleep 35
        done ;;
    *) exit 2 ;;
esac
