#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export STUDY_DIR="$work/study-linked-eight-gpu" MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export MEMSKILL_API_BASE=http://127.0.0.1:18774/v1
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-pre16-duty22-gpu7-qwen2-shared-b2-duty30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
late="$base/late-pairs"
out="$late/cross-window-prefix"
mkdir -p "$out"
check_writer() {
    local pid
    pid=$(ss -H -ltnp 'sport = :18774' | sed -n 's/.*pid=\([0-9]*\),.*/\1/p')
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    python3 - "$pid" "$payload/models/Meta-Llama-3-8B-Instruct" <<'PYTHON'
from pathlib import Path
import sys
args = Path("/proc", sys.argv[1], "cmdline").read_bytes().decode().split("\0")
def option(key):
    return args[args.index(key) + 1]
assert any(Path(arg).name == "serve_low_duty.py" for arg in args)
assert (int(option("--devices")), int(option("--port")), float(option("--duty"))) == (1, 18774, .22)
assert Path(option("--model")).resolve() == Path(sys.argv[2]).resolve()
PYTHON
    nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Fxq "$pid"
}
if [[ ${1:-} == check-writer ]]; then check_writer; exit; fi
if [[ ${1:-} == client ]]; then
    free=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits)
    if ((free < 4352)); then exit 75; fi
    check_writer
    exec bash scripts/experiments/paired_grounding.sh training-cross-prefix --study-forecast "$late/frozen.json" \
        --study-pair-limit "$2" --log-dir "$out/encoder-logs"
fi
server=''
cleanup() { if [[ -n "$server" ]]; then kill -TERM "$server" 2>/dev/null || true; wait "$server" || true; fi; }
trap cleanup EXIT
trap 'exit 143' TERM INT
while [[ ! -s "$out/summary.json" ]]; do
    count=$(find "$out" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)
    if systemctl --user is-active --quiet memskill-251-training-factorial; then
        if ((count > 0)); then sleep 45; continue; fi
        owner=memskill-251-training-factorial
    elif systemctl --user is-active --quiet memskill-251-late-pairs; then
        test -s "$base/source-compression/factorial/summary.json"
        owner=memskill-251-late-pairs
    else
        test -s "$late/outcomes/summary.json"
        owner=''
        if [[ -z "$server" ]]; then
            test -z "$(ss -H -ltn 'sport = :18774')"
            python3 "$guard" run --gpus 7 -- bash scripts/experiments/run_late_pairs_251.sh writer >>"$out/writer.log" 2>&1 &
            server=$!
            while ! curl -fsS --max-time 3 "$MEMSKILL_API_BASE/models" >/dev/null 2>&1; do
                if ! kill -0 "$server" 2>/dev/null; then
                    rc=0; wait "$server" || rc=$?; server=''
                    if [[ $rc == 75 ]]; then sleep 45; break; fi
                    exit 1
                fi
                sleep 4
            done
            [[ -n "$server" ]] || continue
        fi
    fi
    if [[ -z "$(ss -H -ltn 'sport = :18774')" ]]; then
        if [[ -z "$owner" ]]; then exit 1; fi
        sleep 15
        continue
    fi
    rc=0
    python3 "$guard" run --gpus 2 -- bash "$0" client "$((count+1))" >>"$out/client.log" 2>&1 || rc=$?
    if [[ $rc == 75 ]]; then sleep 45; continue; fi
    if ((rc != 0)); then
        if [[ -n "$owner" ]] && ! systemctl --user is-active --quiet "$owner"; then continue; fi
        exit "$rc"
    fi
    # Leave a gap between leases for the original 52-window collection on the same encoder.
    if [[ -n "$owner" && ! -s "$out/summary.json" ]]; then sleep 25; fi
done
