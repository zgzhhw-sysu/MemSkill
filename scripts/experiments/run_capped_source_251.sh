#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export STUDY_DIR="$work/study-linked-eight-gpu" MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export MEMSKILL_API_BASE=http://127.0.0.1:18774/v1
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty22-gpu7-qwen1-shared-b2-duty30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
out="$base/source-compression/capped-source-credit"
args=("$base" "$STUDY_DIR" "$LOCOMO_DATA")
mkdir -p "$out"
mode=${1:-encode}
if [[ $mode == client ]]; then
    free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((free < 4096)); then exit 75; fi
    exec bash scripts/experiments/paired_grounding.sh training-capped-source --study-forecast "$out/registration.json" --log-dir "$out/encoder-logs"
fi
if [[ ! -s "$out/registration.json" ]]; then
    "$MEMSKILL_PYTHON" -m experiments.controller.capped_source_repair register "${args[@]}"
fi
if [[ $mode == policy ]]; then
    last=''
    until [[ -s "$out/policy/joint-repaired/later/summary.json" ]]; do
        state=$(python3 - "$base" "$out" <<'PY'
from pathlib import Path
import sys
b,o=map(Path,sys.argv[1:]);print((o/'complete.json').exists(),len(list((b/'late-pairs/outcomes').glob('conv-*/pair.json'))))
PY
)
        if [[ "$last" != "$state" ]]; then
            python3 "$guard" run --gpus 1 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.capped_source_policy "${args[@]}" >>"$out/policy.log" 2>&1
            last="$state"
        fi
        sleep 45
    done
    exit
fi
[[ $mode == encode || $mode == read ]]
server=''
cleanup() { if [[ -n "$server" ]]; then kill -TERM "$server" 2>/dev/null || true; wait "$server" || true; fi; }
trap cleanup EXIT
trap 'exit 143' TERM INT
until [[ -s "$out/complete.json" ]]; do
    ready=$(python3 - "$out" "$mode" <<'PY'
import json,sys
from pathlib import Path
o=Path(sys.argv[1]); cells=[o/w['name']/bits for w in json.loads((o/'registration.json').read_bytes())['windows'].values() for bits,c in w['cells'].items() if any(c['triggers'])]
if sys.argv[2]=='encode':
    ready=any(not (d/'inputs.json').exists() and (not (d/'suffix-request.json').exists() or (d/'suffix-response.json').exists()) for d in cells)
else:
    ready=all((d/'scores.json').exists() for d in cells) or any((d/'suffix-request.json').exists() and not (d/'suffix-response.json').exists() or (d/'inputs.json').exists() and not (d/'scores.json').exists() for d in cells)
print(int(ready))
PY
)
    if [[ $ready == 0 ]]; then sleep 15; continue; fi
    if [[ $mode == encode ]]; then
        rc=0
        python3 "$guard" run --gpus 1 -- bash "$0" client >>"$out/encoder.log" 2>&1 || rc=$?
        if ((rc!=0 && rc!=75)); then exit "$rc"; fi
        if ((rc==75)); then sleep 35; fi
    else
        if ! bash scripts/experiments/run_cross_prefix_251.sh check-writer; then
            if [[ -n "$server" ]] && ! kill -0 "$server" 2>/dev/null; then rc=0; wait "$server" || rc=$?; server=''; if ((rc!=75 && rc!=0)); then exit "$rc"; fi; fi
            if [[ -z "$server" && -z "$(ss -H -ltn 'sport = :18774')" ]] && ! systemctl --user is-active --quiet memskill-251-{late-pairs,cross-prefix,temporal-update}; then
                python3 "$guard" run --gpus 7 -- bash scripts/experiments/run_late_pairs_251.sh writer >>"$out/server.log" 2>&1 &
                server=$!
            fi
            sleep 20; continue
        fi
        "$MEMSKILL_PYTHON" -u -m experiments.controller.capped_source_repair finish "${args[@]}" >>"$out/reader.log" 2>&1
    fi
done
