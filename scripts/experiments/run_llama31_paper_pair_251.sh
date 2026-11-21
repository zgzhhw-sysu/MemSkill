#!/usr/bin/env bash
# Execute inside the 251 encoder GPU lease. The loopback model server owns its own GPU lease.
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$repo"
run="$work/runs/llama31-paper-v1"
export MEMSKILL_MODEL_MANIFEST="$work/models/Llama-3.1-8B-Instruct/verified-model.json"
export MEMSKILL_BACKEND_ID=llama31-0e9e39f-vllm0102-bf16-eager-greedy-8192-v1
export MEMSKILL_GREEDY_EVAL_CACHE="$run/greedy-cache"
export MEMSKILL_DEVICE=cuda MEMSKILL_ENCODER_DUTY=0.4
api=http://127.0.0.1:18831/v1
while ! curl -fsS --max-time 3 "$api/models" > /dev/null 2>&1; do
    systemctl --user is-active --quiet memskill-llama31-paper-backend-v1.service || exit 2
    sleep 10
done
for dataset in locomo longmemeval; do
    data="$LOCOMO_DATA"
    if [[ "$dataset" == longmemeval ]]; then data="$LONGMEMEVAL_DATA"; fi
    "$MEMSKILL_PYTHON" -u -m experiments.designer.paper_date_pair \
        --dataset "$dataset" --data-file "$data" --load-checkpoint "$(dirname "$LOCOMO_DATA")/conversation_controller.pt" \
        --api --api-base "$api" --api-key local-only --device cuda --disable-flash-attn \
        --state-encoder "$MEMSKILL_QWEN_EMBEDDING_PATH" --op-encoder "$MEMSKILL_QWEN_EMBEDDING_PATH" \
        --encode-batch-size 2 --reward-metric f1 --out-file "$run/$dataset/comparison.json" \
        --save-dir "$run/$dataset/checkpoints" --log-dir "$run/$dataset/logs"
done
systemctl --user stop memskill-llama31-paper-backend-v1.service
