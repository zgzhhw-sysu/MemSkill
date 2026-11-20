#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
dest="$base/late-pairs"
d="$dest/step-mixture"
args=("$MEMSKILL_PYTHON" -m experiments.controller.audit_late_pairs "$work/study-linked-eight-gpu" "$LOCOMO_DATA" "$base/source-compression" "$dest")
previous_full=-1
previous_pair=0
while true; do
    full=$(find "$base/source-compression/factorial" -mindepth 2 -maxdepth 2 -name measurement.json | wc -l)
    n=0
    if [[ -d "$dest/outcomes" ]]; then
        n=$(find "$dest/outcomes" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)
    fi
    if [[ "$full" != "$previous_full" ]]; then
        python3 "$guard" run --gpus 0 -- "${args[@]}"
        previous_full=$full
    fi
    if [[ "$n" != "$previous_pair" ]]; then
        python3 "$guard" run --gpus 0 -- "${args[@]}" --randomized
        previous_pair=$n
    fi
    if [[ -s "$d/summary.json" && -s "$base/source-compression/factorial/summary.json" ]]; then exit 0; fi
    if ! systemctl --user is-active --quiet memskill-251-late-pairs; then test -s "$dest/outcomes/summary.json" || exit 1; fi
    sleep 60
done
