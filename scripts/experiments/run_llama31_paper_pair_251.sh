#!/usr/bin/env bash
# Execute inside the 251 encoder GPU lease. The loopback model server owns its own GPU lease.
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "${MEMSKILL_EVAL_REPO:-$repo}"
baseline_run="$work/runs/llama31-paper-v1"
run="$baseline_run"
export MEMSKILL_GREEDY_EVAL_CACHE="$baseline_run/greedy-cache"
export MEMSKILL_MODEL_MANIFEST="$work/models/Llama-3.1-8B-Instruct/verified-model.json"
export MEMSKILL_BACKEND_ID=llama31-0e9e39f-vllm0102-bf16-tp2-eager-duty80-8192-v1
export MEMSKILL_PAPER_NATIVE_ONLY=1
export MEMSKILL_PAPER_SPLIT=test
unset MEMSKILL_LONGMEMEVAL_LAUNCHER MEMSKILL_STATE_CREDIT_BASE MEMSKILL_SWAP_HISTORY MEMSKILL_SWAP_SPAN MEMSKILL_SWAP_REPEAT MEMSKILL_NATIVE_TRAIN_ROOT
report=baseline.json
mode=${1:-baseline}
datasets=(locomo longmemeval)
encoder_batch=2
case "$mode" in
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
    longmemeval-launcher)
        datasets=(longmemeval)
        encoder_batch=64
        run="$baseline_run-$mode"
        export MEMSKILL_LONGMEMEVAL_LAUNCHER=1 MEMSKILL_QWEN_PREFILL_ONLY=1
        export MEMSKILL_MEMORY_SEARCH=cuda-skylakex PYTORCH_ALLOC_CONF=expandable_segments:True
        export MEMSKILL_GREEDY_EVAL_CACHE="$run/greedy-cache"
        ;;
    locomo-batch64|locomo-train|locomo-validation|locomo-train-swap|locomo-credit-validation)
        datasets=(locomo)
        encoder_batch=64
        run="$baseline_run-$mode"
        if [[ "$mode" == locomo-train-swap ]]; then
            if [[ $# -lt 3 || $# -gt 4 || ! ${2:-} =~ ^conv-[0-9]+$ || ! ${3:-} =~ ^(0|[1-9][0-9]*)$
                  || ( $# == 4 && ! $4 =~ ^[1-9][0-9]*$ ) ]]; then
                echo "Usage: locomo-train-swap conv-ID zero-based-span [positive-repeat]" >&2
                exit 2
            fi
            for baseline in "$baseline_run-locomo-batch64/legacy-rank-audit/upstream-inference-batch64.json" \
                            "$baseline_run/longmemeval/baseline.json"; do
                if [[ ! -s "$baseline" ]]; then echo "Waiting for original baseline: $baseline" >&2; exit 1; fi
            done
            export MEMSKILL_PAPER_SPLIT=train MEMSKILL_SWAP_HISTORY="$2" MEMSKILL_SWAP_SPAN="$3"
            export MEMSKILL_NATIVE_TRAIN_ROOT="$baseline_run-locomo-train/locomo"
            export MEMSKILL_GREEDY_EVAL_CACHE="$baseline_run-locomo-train/greedy-cache"
            run="$run/$2/span-$3"
            if [[ $# == 4 ]]; then
                export MEMSKILL_SWAP_REPEAT="$4"
                run="$run/repeats/$4"
            fi
        elif [[ "$mode" == locomo-credit-validation ]]; then
            test -s "$baseline_run/paired-baselines-verified.json"
            test -s "$baseline_run-locomo-train-swap/panel-summary.json"
            export MEMSKILL_PAPER_SPLIT=validation MEMSKILL_STATE_CREDIT_BASE="$baseline_run"
            export MEMSKILL_GREEDY_EVAL_CACHE="$baseline_run-locomo-validation/greedy-cache"
            report=comparison.json
        elif [[ "$mode" != locomo-batch64 ]]; then
            export MEMSKILL_PAPER_SPLIT="${mode#locomo-}" MEMSKILL_GREEDY_EVAL_CACHE="$run/greedy-cache"
        fi
        export MEMSKILL_QWEN_PREFILL_ONLY=1 MEMSKILL_MEMORY_SEARCH=cuda-skylakex MEMSKILL_TRACE_NATIVE_POLICY=1
        export PYTORCH_ALLOC_CONF=expandable_segments:True
        ;;
    paired) export MEMSKILL_PAPER_NATIVE_ONLY=0; report=comparison.json ;;
    *) echo "Expected backend, baseline, longmemeval-launcher, locomo-batch64, locomo-train, locomo-validation, locomo-train-swap, locomo-credit-validation or paired" >&2; exit 2 ;;
esac
export MEMSKILL_DEVICE=cuda MEMSKILL_ENCODER_DUTY=0.8 MEMSKILL_MEMORY_SEARCH="${MEMSKILL_MEMORY_SEARCH:-cuda}"
if [[ "$MEMSKILL_MEMORY_SEARCH" == cuda-skylakex ]]; then
    if [[ "$mode" != locomo-* && "$mode" != longmemeval-launcher ]]; then run="$baseline_run-cuda-skylakex"; fi
    export CC="${CC:-$HOME/.conda/envs/agemem/bin/x86_64-conda-linux-gnu-cc}" OPENBLAS_NUM_THREADS=1
fi
api=http://127.0.0.1:18831/v1
while ! curl -fsS --max-time 3 "$api/models" > /dev/null 2>&1; do
    systemctl --user is-active --quiet memskill-llama31-baseline-tp2.service || exit 2
    sleep 10
done
for dataset in "${datasets[@]}"; do
    data="$LOCOMO_DATA"
    if [[ "$dataset" == longmemeval ]]; then data="$LONGMEMEVAL_DATA"; fi
    "$MEMSKILL_PYTHON" -u -m experiments.designer.paper_date_pair \
        --dataset "$dataset" --data-file "$data" --load-checkpoint "$(dirname "$LOCOMO_DATA")/conversation_controller.pt" \
        --api --api-base "$api" --api-key local-only --device cuda --disable-flash-attn \
        --state-encoder "$MEMSKILL_QWEN_EMBEDDING_PATH" --op-encoder "$MEMSKILL_QWEN_EMBEDDING_PATH" \
        --encode-batch-size "$encoder_batch" --reward-metric f1 --out-file "$run/$dataset/$report" \
        --save-dir "$run/$dataset/checkpoints" --log-dir "$run/$dataset/logs"
done
if [[ "$MEMSKILL_PAPER_SPLIT" != test || "$mode" == longmemeval-launcher ]]; then
    exit 0
elif [[ "$mode" == locomo-batch64 ]]; then
    run="$baseline_run"
elif [[ ! -s "$baseline_run-locomo-batch64/locomo/baseline.json" ]] &&
    systemctl --user is-active --quiet memskill-llama31-locomo-batch64.service; then
    exit 0
fi
if [[ -s "$run/locomo/$report" && -s "$run/longmemeval/$report" ]]; then
    systemctl --user stop memskill-llama31-baseline-tp2.service
fi
