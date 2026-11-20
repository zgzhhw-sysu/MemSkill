#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MEMSKILL_API_BASE=http://127.0.0.1:18774/v1
base="$work/runs/paired-251-linked-eight"
out="$base/source-compression/reader-slot-credit"
study="$work/study-linked-eight-gpu"
mkdir -p "$out"
role=${1:-reader}
if [[ $role == reader || $role == witness ]]; then
    if [[ $role == witness ]]; then
        [[ ! -s "$out/single-rank/summary.json" ]] || exit 0
        reader=(experiments.controller.reader_slot_witness "$base" "$LOCOMO_DATA")
    else
        [[ ! -s "$out/readers-complete.json" ]] || exit 0
        reader=(experiments.controller.reader_slot_credit "$base" "$study" "$LOCOMO_DATA")
    fi
    "$MEMSKILL_PYTHON" -m "${reader[@]}"
    if bash scripts/experiments/run_cross_prefix_251.sh check-writer; then
        exec "$MEMSKILL_PYTHON" -u -m "${reader[@]}" --read
    fi
    if [[ -z "$(ss -H -ltn 'sport = :18774')" ]] &&
       ! systemctl --user is-active --quiet memskill-251-{late-pairs,cross-prefix,first-evidence,temporal-update}; then
        python3 "$guard" run --gpus 7 -- bash scripts/experiments/run_late_pairs_251.sh writer &
        server=$!
        trap 'kill -TERM "$server" 2>/dev/null || true; wait "$server" || true' EXIT
        until bash scripts/experiments/run_cross_prefix_251.sh check-writer; do
            kill -0 "$server" || exit 75
            sleep 10
        done
        "$MEMSKILL_PYTHON" -u -m "${reader[@]}" --read
        exit 0
    fi
    exit 75
fi
[[ $role == policy || $role == prefix ]]
gpu=${MEMSKILL_SLOT_POLICY_GPU:-0}
[[ $gpu == 0 || $gpu == 1 ]]
observed="$base/late-pairs/outcomes"
panel=later
args=()
if [[ $role == prefix ]]; then
    gpu=1
    observed="$base/late-pairs/cross-window-prefix"
    panel=long-prefix
    args=(--long-prefix)
fi
last=''
until [[ -s "$out/complete/$panel/summary.json" ]]; do
    pairs=$(find "$observed" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)
    state="$pairs-$(test -s "$out/readers-complete.json" && echo complete || echo partial)"
    if [[ -s "$out/cohort.json" && $state != "$last" ]]; then
        "$MEMSKILL_PYTHON" -m experiments.controller.reader_slot_policy "$base" "$LOCOMO_DATA" --freeze
        python3 "$guard" run --gpus "$gpu" -- "$MEMSKILL_PYTHON" -m experiments.controller.reader_slot_policy "$base" "$LOCOMO_DATA" "${args[@]}"
        last=$state
    fi
    sleep 60
done
