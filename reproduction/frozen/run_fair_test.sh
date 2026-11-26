#!/usr/bin/env bash
set -euo pipefail
root=/home/lixian/Documents/研究生大论文/review-stage/hybrid-controller-20261126
export MEMSKILL_MODEL_MANIFEST="$root/snapshot/models/Llama-3.1-8B-Instruct/verified-model.json"
export MEMSKILL_BACKEND_ID=llama31-0e9e39f-vllm0102-bf16-tp2-eager-duty80-8192-v1
export MEMSKILL_DEVICE=cuda MEMSKILL_ENCODER_DUTY=0.8 MEMSKILL_QWEN_PREFILL_ONLY=1
export MEMSKILL_MEMORY_SEARCH=cuda-skylakex MEMSKILL_TRACE_NATIVE_POLICY=1
export MEMSKILL_API_BASE=http://127.0.0.1:18931/v1 MEMSKILL_API_TIMEOUT_S=900
export MEMSKILL_QWEN_EMBEDDING_PATH=/home/lixian/Documents/AgentMemory/models/Qwen3-Embedding-0.6B
export TIKTOKEN_CACHE_DIR="$root/assets/tiktoken"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 WANDB_MODE=disabled
export CC=/home/lixian/.conda/envs/gfmrag/bin/x86_64-conda-linux-gnu-cc
export MEMSKILL_PAPER_SPLIT=test MEMSKILL_PAPER_NATIVE_ONLY=1
mode="$1"; arm="$2"; sid="$3"
export FAIR_TEST_ARM="$arm" FAIR_TEST_HISTORY="$sid"
cd "$root/snapshot/run-linked-questions-v1"
export PYTHONPATH="$PWD"
py=/home/lixian/Documents/AgentMemory/.venv-judge24b/bin/python
if [[ "$mode" == prepare ]]; then exec "$py" -u "$root/prepare_fair_test.py"; fi
if [[ "$mode" == mixed ]]; then exec "$py" -u "$root/fair_test_mixed.py" "$sid"; fi
if [[ "$mode" == score ]]; then exec "$py" -u "$root/score_fair_test.py" "$sid"; fi
[[ "$mode" == writer ]]
exec "$py" -u "$root/fair_test_writer.py" --dataset locomo --data-file "$root/assets/locomo10.json" --load-checkpoint "$root/assets/conversation_controller.pt" --api --api-base "$MEMSKILL_API_BASE" --api-key local-only --device cuda --disable-flash-attn --state-encoder "$MEMSKILL_QWEN_EMBEDDING_PATH" --op-encoder "$MEMSKILL_QWEN_EMBEDDING_PATH" --encode-batch-size 64 --reward-metric f1 --out-file "$root/fair-test-three-v1/writers/$arm/report.json" --save-dir "$root/fair-test-three-v1/writers/$arm/checkpoints" --log-dir "$root/fair-test-three-v1/writers/$arm/logs"
