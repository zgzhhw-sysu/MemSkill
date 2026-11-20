#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
parent="$base/source-compression/witness-passages"
out="$parent/evidence-advantage"
mkdir -p "$out"
count() { if [[ -d $1 ]]; then find "$1" -mindepth 2 -maxdepth 2 -name "$2" | wc -l; else echo 0; fi; }
role=${1:-local}
case "$role" in
    local) module=passage_advantage; done_file="$out/witness/decision.json" ;;
    case) module=passage_case_reliability; done_file="$out/case-reliability/witness/decision.json" ;;
    prefix)
        module=passage_evidence_prefix; done_file="$out/witness/long-prefix/summary.json"
        export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
        until [[ -s "$out/prefix-sources/complete.json" ]]; do
            rc=0
            python3 "$guard" run --gpus 4 -- "$MEMSKILL_PYTHON" -m experiments.controller.passage_prefix_sources "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" >>"$out/prefix-encoder.log" 2>&1 || rc=$?
            if ((rc==75)); then sleep 45; elif ((rc!=0)); then exit "$rc"; fi
        done ;;
    *) exit 2 ;;
esac
last=''
until [[ -s "$done_file" && ( $role != case || -s "$out/case-reliability/witness/long-prefix/summary.json" ) ]]; do
    if [[ $role == prefix ]]; then
        if [[ ! -s "$out/group/forecast.json" ]]; then sleep 45; continue; fi
        state="$(count "$base/late-pairs/cross-window-prefix" pair.json)-$(test -s "$out/witness/forecast.json" && echo witness || echo group)"
    elif [[ $role == case ]]; then
        if [[ ! -s "$out/group/early/summary.json" ]]; then sleep 45; continue; fi
        state="$(count "$base/late-pairs/outcomes" pair.json)-$(count "$base/late-pairs/donor-factorial" complete.json)-$(count "$parent/four-paths/cells" complete.json)-$(count "$base/late-pairs/cross-window-prefix" pair.json)-$(test -s "$out/witness/forecast.json" && echo witness || echo group)-$(test -s "$out/prefix-sources/complete.json" && echo ready || echo waiting)"
    else
        if [[ ! -s "$parent/group/forecast.json" ]]; then sleep 45; continue; fi
        state="$(count "$base/late-pairs/outcomes" pair.json)-$(count "$base/late-pairs/donor-factorial" complete.json)-$(count "$parent/four-paths/cells" complete.json)-$(test -s "$parent/witness/forecast.json" && echo witness || echo group)"
    fi
    if [[ $state != "$last" ]]; then
        rc=0
        python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m "experiments.controller.$module" "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" >>"$out/$role.log" 2>&1 || rc=$?
        if ((rc==0)); then last=$state; elif ((rc!=75)); then exit "$rc"; fi
    fi
    sleep 45
done
