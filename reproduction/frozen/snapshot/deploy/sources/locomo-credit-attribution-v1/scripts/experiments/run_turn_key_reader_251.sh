#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export MEMSKILL_API_BASE=http://127.0.0.1:18774/v1
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty22-gpu7-qwen1-shared-b2-duty30
export STUDY_DIR="$work/study-linked-eight-gpu" MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight/source-compression"
out="$base/turn-keys"
if [[ ${1:-} == prepare ]]; then
    exec bash scripts/experiments/paired_grounding.sh training-turn-keys --study-forecast "$base/frozen.json" --log-dir "$out/encoder-logs"
fi
mkdir -p "$out/reader"
prepare_windows() {
    while [[ ! -s "$out/reader-summary.json" ]]; do
        if [[ ! -s "$out/screen.json" ]]; then sleep 15; continue; fi
        local ready prepared rc=0
        ready=$(find "$base/factorial" -mindepth 2 -maxdepth 2 -name measurement.json | wc -l)
        prepared=$(find "$out/reader" -mindepth 2 -maxdepth 2 -name jobs.json | wc -l)
        if ((prepared == 12)); then return 0; fi
        if ((ready <= prepared)); then sleep 30; continue; fi
        python3 "$guard" run --gpus 1 -- bash "$0" prepare >>"$out/prepare.log" 2>&1 || rc=$?
        if [[ $rc == 75 ]]; then sleep 30; continue; fi
        if ((rc != 0)); then return "$rc"; fi
    done
}

score_windows() {
    while [[ ! -s "$out/reader-summary.json" ]]; do
        local prepared answered rc=0
        prepared=$(find "$out/reader" -mindepth 2 -maxdepth 2 -name jobs.json | wc -l)
        answered=$(find "$out/reader" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)
        if ((answered >= prepared && answered < 12)); then sleep 15; continue; fi
        if ! bash scripts/experiments/run_cross_prefix_251.sh check-writer; then sleep 20; continue; fi
        "$MEMSKILL_PYTHON" -u -m experiments.controller.turn_key_reader "$out" --limit "$((answered+1))" >>"$out/reader.log" 2>&1 || rc=$?
        if ((rc != 0)); then
            if ! bash scripts/experiments/run_cross_prefix_251.sh check-writer; then sleep 20; continue; fi
            return "$rc"
        fi
        # A later four-path window can finish before the next registered one.
        if ((answered == $(find "$out/reader" -mindepth 2 -maxdepth 2 -name pair.json | wc -l))); then sleep 15; fi
    done
}

# Each job receipt is atomically published. GPU 7 can answer one window while GPU 1 prepares the next.
prepare_windows &
preparer=$!
score_windows &
scorer=$!
trap 'kill "$preparer" "$scorer" 2>/dev/null || true' EXIT
wait -n "$preparer" "$scorer"
wait "$preparer"
wait "$scorer"
trap - EXIT
