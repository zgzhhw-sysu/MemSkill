#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export MEMSKILL_BACKEND_ID=251-llama3-bf16-pre16-duty22-gpu7-case-credit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
dest="$base/source-compression/case-credit"
mkdir -p "$dest/joint"
if [[ ${1:-} == writer ]]; then
    python3 "$guard" run --gpus 7 -- bash scripts/experiments/run_late_pairs_251.sh writer >>"$dest/writer.log" 2>&1 &
    owner=$!
    trap 'kill -TERM "$owner" 2>/dev/null || true; wait "$owner" || true' EXIT
    trap 'exit 143' TERM INT
    while kill -0 "$owner" 2>/dev/null; do
        if [[ -e "$dest/votes.done" && -e "$dest/joint/votes.done" && -z "$(ss -Hnt state established 'sport = :18774')" ]]; then exit 0; fi
        sleep 40
    done
    wait "$owner"
    exit "$?"
fi
module=experiments.controller.case_credit
output="$dest"
if [[ ${1:-} == joint ]]; then
    module=experiments.controller.case_credit_joint_forecast
    output="$dest/joint"
    until [[ -s "$dest/prompts.json" ]]; do sleep 15; done
fi
until bash scripts/experiments/run_cross_prefix_251.sh check-writer; do
    if [[ -z "$(ss -H -ltn 'sport = :18774')" ]] &&
       ! systemctl --user is-active --quiet memskill-251-late-pairs &&
       ! systemctl --user is-active --quiet memskill-251-cross-prefix &&
       ! systemctl --user is-active --quiet memskill-251-case-credit-writer; then
        systemd-run --user --collect --unit=memskill-251-case-credit-writer --property=Restart=on-failure --property=RestartSec=45 \
            bash "$PWD/scripts/experiments/run_case_credit_251.sh" writer || systemctl --user is-active --quiet memskill-251-case-credit-writer
    fi
    sleep 25
done
"$MEMSKILL_PYTHON" -u -m "$module" "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA" >>"$output/votes.log" 2>&1
touch "$output/votes.done"
