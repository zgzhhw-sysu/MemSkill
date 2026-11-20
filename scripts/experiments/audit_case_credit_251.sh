#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
src="$base/source-compression"
out="$src/case-credit"
mkdir -p "$out"
last=''
while [[ ! -s "$out/later/summary.json" || ! -s "$out/factorial/summary.json" || ! -s "$out/joint/later/summary.json" || ! -s "$out/joint/factorial/summary.json" ]]; do
    if [[ ! -s "$out/prompts.json" || ! -d "$out/choices" ]]; then sleep 15; continue; fi
    n=$(find "$out/choices" -name '*.json' | wc -l)
    m=$(find "$src/retrieval-competition/pairs" -name '*.json' | wc -l)
    r=$(find "$out/responses" -name '*.json' | wc -l)
    j=0
    if [[ -d "$out/joint/choices" ]]; then j=$(find "$out/joint/choices" -name '*.json' | wc -l); fi
    if ((n == 37)); then touch "$out/votes.done"; fi
    if ((j == 91)); then touch "$out/joint/votes.done"; fi
    checkpoint="$n-$m-$((r/3))-$j"
    if [[ "$last" == "$checkpoint" ]]; then sleep 30; continue; fi
    python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.audit_case_credit "$base" >>"$out/audit.log" 2>&1
    if ((j > 0)); then
        python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.audit_case_credit_joint "$base" >>"$out/joint/audit.log" 2>&1
    fi
    last="$checkpoint"
done
