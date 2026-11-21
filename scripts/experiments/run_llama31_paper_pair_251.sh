#!/usr/bin/env bash
# Execute inside the 251 encoder GPU lease. The loopback model server owns its own GPU lease.
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "${MEMSKILL_EVAL_REPO:-$repo}"
run="$work/runs/llama31-paper-v1"
export MEMSKILL_MODEL_MANIFEST="$work/models/Llama-3.1-8B-Instruct/verified-model.json"
export MEMSKILL_BACKEND_ID=llama31-0e9e39f-vllm0102-bf16-tp2-eager-duty80-8192-v1
export MEMSKILL_PAPER_NATIVE_ONLY=1
report=baseline.json
case "${1:-baseline}" in
    backend)
        test -s "$MEMSKILL_MODEL_MANIFEST"
        export MEMSKILL_VLLM_DUTY=.80 VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
        export PYTHONPATH="$PWD/scripts/experiments${PYTHONPATH:+:$PYTHONPATH}" NCCL_P2P_DISABLE=1
        exec "$HOME/.conda/envs/agemem/bin/python" -m vllm.entrypoints.openai.api_server \
            --model "$(dirname "$MEMSKILL_MODEL_MANIFEST")" --served-model-name meta-llama/Llama-3.1-8B-Instruct \
            --host 127.0.0.1 --port 18831 --dtype bfloat16 --seed 42 --enforce-eager --max-model-len 8192 \
            --tensor-parallel-size 2 --gpu-memory-utilization .43 --max-num-seqs 4 --max-num-batched-tokens 256 \
            --disable-custom-all-reduce --disable-log-requests --worker-cls serve_low_duty.DutyLimitedWorker ;;
    baseline) ;;
    paired) export MEMSKILL_PAPER_NATIVE_ONLY=0; report=comparison.json ;;
    *) echo "Expected backend, baseline or paired" >&2; exit 2 ;;
esac
export MEMSKILL_GREEDY_EVAL_CACHE="$run/greedy-cache"
export MEMSKILL_DEVICE=cuda MEMSKILL_ENCODER_DUTY=0.8
api=http://127.0.0.1:18831/v1
while ! curl -fsS --max-time 3 "$api/models" > /dev/null 2>&1; do
    systemctl --user is-active --quiet memskill-llama31-baseline-tp2.service || exit 2
    sleep 10
done
for dataset in locomo longmemeval; do
    data="$LOCOMO_DATA"
    if [[ "$dataset" == longmemeval ]]; then data="$LONGMEMEVAL_DATA"; fi
    "$MEMSKILL_PYTHON" -u -m experiments.designer.paper_date_pair \
        --dataset "$dataset" --data-file "$data" --load-checkpoint "$(dirname "$LOCOMO_DATA")/conversation_controller.pt" \
        --api --api-base "$api" --api-key local-only --device cuda --disable-flash-attn \
        --state-encoder "$MEMSKILL_QWEN_EMBEDDING_PATH" --op-encoder "$MEMSKILL_QWEN_EMBEDDING_PATH" \
        --encode-batch-size 2 --reward-metric f1 --out-file "$run/$dataset/$report" \
        --save-dir "$run/$dataset/checkpoints" --log-dir "$run/$dataset/logs"
done
if [[ -s "$run/locomo/$report" && -s "$run/longmemeval/$report" ]]; then
    systemctl --user stop memskill-llama31-baseline-tp2.service
fi
