#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." || exit 1
phase=${1:?Pass train, probe-omitted, validation, test or transfer}
shift
: "${STUDY_DIR:?Set a directory for this experiment}"
: "${LOCOMO_DATA:?Set the path to locomo10.json}"
: "${CONTROLLER_CHECKPOINT:?Set the native conversation Controller checkpoint}"
: "${MEMSKILL_QWEN_EMBEDDING_PATH:?Set the local Qwen3-Embedding-0.6B model directory}"
if [[ $phase == test || $phase == transfer ]] && [[ -f "$STUDY_DIR/require-selective-validation" ]]; then
    "${MEMSKILL_PYTHON:-python}" -m experiments.controller.audit_paired "$STUDY_DIR" "$LOCOMO_DATA" validation --select-for-test
fi
module=experiments.controller.paired_grounding
dataset=locomo
data_file=$LOCOMO_DATA
phase_args=(--study-phase "$phase" --span-credit-audit)
if [[ $phase == transfer ]]; then
    module=experiments.controller.transfer_grounding
    dataset=longmemeval
    data_file=${LONGMEMEVAL_DATA:?Set the path to longmemeval_s_cleaned.json}
    phase_args=()
    writer_model=${MEMSKILL_WRITER_TOKENIZER_PATH:-"$(dirname -- "$MEMSKILL_QWEN_EMBEDDING_PATH")/Meta-Llama-3-8B-Instruct"}
    phase_args+=(--study-writer-tokenizer "$writer_model")
fi
exec "${MEMSKILL_PYTHON:-python}" -u -m "$module" "${phase_args[@]}" \
    --study-dir "$STUDY_DIR" --dataset "$dataset" --data-file "$data_file" \
    --load-checkpoint "$CONTROLLER_CHECKPOINT" --model llama3-8b --api --api-base "${MEMSKILL_API_BASE:?}" \
    --api-key "${MEMSKILL_API_KEY:-local-only}" --state-encoder "$MEMSKILL_QWEN_EMBEDDING_PATH" \
    --op-encoder "$MEMSKILL_QWEN_EMBEDDING_PATH" --retriever qwen3-embedding-0.6b --disable-flash-attn \
    --skip-load-snapshot-manager --session-mode full-session --action-top-k 3 --mem-top-k 5 --mem-top-k-eval 5 \
    --device "${MEMSKILL_DEVICE:-cpu}" --inference-workers 1 --inference-session-workers 1 --encode-batch-size 2 \
    --eval-f1-only --reward-metric f1 --seed 42 --span-credit-spans 2 --span-credit-questions 4 \
    --max-new-tokens 256 --temperature 0 --round 1 --trace-dir "$STUDY_DIR/traces" \
    --log-dir "$STUDY_DIR/logs/$phase" --save-dir "$STUDY_DIR/checkpoints" --out-file "$STUDY_DIR/result.json" "$@"
