#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export STUDY_DIR="$work/study-linked-eight-gpu" MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export MEMSKILL_API_BASE=http://127.0.0.1:18774/v1 MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty22-gpu7-qwen2-shared-b2-duty30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
dest="$base/late-pairs"
if [[ ${1:-} == writer ]]; then
    free=$(nvidia-smi -i 7 --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((free<18432)); then exit 75; fi
    exec "$MEMSKILL_PYTHON" -u scripts/experiments/serve_low_duty.py --devices 1 --reserve-mib 2048 --duty .22 --port 18774 --model "$payload/models/Meta-Llama-3-8B-Instruct"
fi
if [[ ${1:-} == client ]]; then
    free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((free<4352)); then exit 75; fi
    shift
    exec bash scripts/experiments/paired_grounding.sh training-late-pairs --study-forecast "$dest/frozen.json" --log-dir "$dest/encoder-logs" "$@"
fi
client() {
    while true; do
        rc=0; python3 "$guard" run --gpus 2 -- bash "$0" client "$@" >>"$dest/client.log" 2>&1 || rc=$?
        if [[ $rc == 75 ]]; then sleep 45; continue; fi
        return "$rc"
    done
}
if [[ ! -s "$dest/frozen.json" ]]; then
    prepare=("$MEMSKILL_PYTHON" -m experiments.controller.late_pair_policy)
    inputs=("$work/study-equal-sources-gpu" "$STUDY_DIR" "$LOCOMO_DATA" "$base/source-compression" "$dest")
    for phase in features freeze; do
        python3 "$guard" run --gpus 0 -- "${prepare[@]}" "$phase" "${inputs[@]}"
    done
fi
if [[ ! -s "$dest/step-mixture/forecast.json" ]]; then
    python3 "$guard" run --gpus 0 -- "$MEMSKILL_PYTHON" -m experiments.controller.step_mixture "$base/source-compression" "$dest"
fi
if [[ ! -s "$dest/step-mixture/deployment.json" ]]; then
    python3 "$guard" run --gpus 0 -- "$MEMSKILL_PYTHON" -m experiments.controller.step_mixture "$base/source-compression" "$dest" --full-fit "$work/study-equal-sources-gpu" "$LOCOMO_DATA"
fi
if [[ -s "$dest/outcomes/summary.json" ]]; then exit 0; fi
if [[ ! -s "$dest/outcomes/conv-26_s9/pair.json" ]] && systemctl --user is-active --quiet memskill-251-training-factorial; then
    client --study-pair-limit 1 || { systemctl --user is-active --quiet memskill-251-training-factorial && exit 1; }
fi
while systemctl --user is-active --quiet memskill-251-training-factorial; do sleep 45; done
test -s "$base/source-compression/factorial/summary.json"
server=''
cleanup() { if [[ -n "$server" ]]; then kill -TERM "$server" 2>/dev/null || true; wait "$server" || true; fi; }
trap cleanup EXIT
trap 'exit 143' TERM INT
while true; do
    test -z "$(ss -H -ltn 'sport = :18774')"
    python3 "$guard" run --gpus 7 -- bash "$0" writer >>"$dest/writer.log" 2>&1 &
    server=$!
    until curl -fsS --max-time 3 "$MEMSKILL_API_BASE/models" >/dev/null 2>&1; do
        if ! kill -0 "$server" 2>/dev/null; then
            rc=0; wait "$server" || rc=$?; server=''
            if [[ $rc == 75 ]]; then sleep 45; break; fi
            exit 1
        fi
        sleep 4
    done
    [[ -z "$server" ]] || break
done
windows=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["assignments"]))' "$dest/cohort.json")
for ((i=2;i<=windows;++i)); do
    client --study-pair-limit "$i"
done
test -s "$dest/outcomes/summary.json"
