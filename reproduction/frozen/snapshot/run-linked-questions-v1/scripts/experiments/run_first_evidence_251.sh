#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export STUDY_DIR="$work/study-linked-eight-gpu" MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export MEMSKILL_API_BASE=http://127.0.0.1:18774/v1
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty22-gpu7-qwen2-shared-b2-duty30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
out="$base/late-pairs/first-evidence"
mkdir -p "$out"
if [[ ${1:-measure} == client ]]; then
    free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( free < 4352 )); then exit 75; fi
    bash scripts/experiments/run_cross_prefix_251.sh check-writer
    exec bash scripts/experiments/paired_grounding.sh training-first-evidence --study-forecast "$out/registration.json" --log-dir "$out/encoder-logs"
fi
if [[ ${1:-measure} == interaction ]]; then
    d="$base/source-compression/prefix-interaction"
    mkdir -p "$d"
    last=''
    last_prefix=''
    until [[ -s "$d/donor-12.json" && $last_prefix == 99 ]]; do
        count=$(find "$base/late-pairs/donor-factorial" -mindepth 2 -maxdepth 2 -name complete.json | wc -l)
        if [[ "$last" != "$count" ]]; then
            python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.prefix_interaction "$base" >>"$d/run.log" 2>&1
            last=$count
        fi
        prefixes=$(find "$base/late-pairs/cross-window-prefix" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)
        if [[ "$last_prefix" != "$prefixes" ]]; then
            python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -m experiments.controller.first_evidence_credit --forecasts "$base" >>"$d/forecasts.log" 2>&1
            last_prefix=$prefixes
        fi
        sleep 45
    done
    exit
fi
"$MEMSKILL_PYTHON" -m experiments.controller.first_evidence_credit "$base"
role=${1:-measure}
[[ "$role" == measure || "$role" == prefill ]]
server=''
cleanup() { if [[ -n "$server" ]]; then kill -TERM "$server" 2>/dev/null || true; wait "$server" || true; fi; }
trap cleanup EXIT
trap 'exit 143' TERM INT
until [[ -s "$out/summary.json" ]]; do
    ready=$(python3 - "$out" "$role" <<'PY'
import json,sys
from pathlib import Path
o=Path(sys.argv[1])
groups=json.loads((o/'registration.json').read_bytes())['groups']
def cached(g):
    return (o/'first-writes'/f"{g['dialogue']}_s{g['first']:02d}"/'response.json').is_file()
if sys.argv[2]=='prefill':
    pending=[g for g in groups if not (o/'first-writes'/f"{g['dialogue']}_s{g['first']:02d}"/'writes.json').is_file()]
    print(2 if not pending else int(any((o.parent/'cross-window-prefix'/g['endpoint']/'pair.json').is_file() for g in pending)))
else:
    pending=[g for g in groups if not (o/g['name']/'pair.json').is_file()]
    print(int(not pending or any(cached(g) and (o.parent/'cross-window-prefix'/g['endpoint']/'pair.json').is_file() for g in pending)))
PY
)
    if [[ $ready == 2 ]]; then exit; fi
    if [[ $ready == 0 ]]; then sleep 45; continue; fi
    if ! bash scripts/experiments/run_cross_prefix_251.sh check-writer; then
        if [[ -n "$server" ]] && ! kill -0 "$server" 2>/dev/null; then rc=0; wait "$server" || rc=$?; server=''; if ((rc!=75 && rc!=0)); then exit "$rc"; fi; fi
        if [[ -z "$server" && -z "$(ss -H -ltn 'sport = :18774')" ]] && ! systemctl --user is-active --quiet memskill-251-{late-pairs,cross-prefix,temporal-update}; then
            python3 "$guard" run --gpus 7 -- bash scripts/experiments/run_late_pairs_251.sh writer >>"$out/server.log" 2>&1 &
            server=$!
        fi
        sleep 20; continue
    fi
    if [[ $role == prefill ]]; then
        "$MEMSKILL_PYTHON" -m experiments.controller.first_evidence_prefill "$base" "$STUDY_DIR" >>"$out/prefill.log" 2>&1
        continue
    fi
    rc=0
    python3 "$guard" run --gpus 2 -- bash "$0" client >>"$out/client.log" 2>&1 || rc=$?
    if ((rc!=0 && rc!=75)); then exit "$rc"; fi
    sleep 60
done
