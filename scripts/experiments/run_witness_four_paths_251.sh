#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export STUDY_DIR="$work/study-linked-eight-gpu" MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export MEMSKILL_API_BASE=http://127.0.0.1:18773/v1
export MEMSKILL_BACKEND_ID=251-hf-llama3-bf16-gpu56-duty28-qwen4-b2-duty30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
out="$base/source-compression/witness-passages/four-paths"
model="$payload/models/Meta-Llama-3-8B-Instruct"
program=("$MEMSKILL_PYTHON" -m experiments.controller.witness_passage_factorial "$base" "$STUDY_DIR")
check_writer() {
    local pid gpu
    pid=$(ss -H -ltnp 'sport = :18773' | sed -n 's/.*pid=\([0-9]*\),.*/\1/p')
    [[ $pid =~ ^[0-9]+$ ]] || return 1
    python3 - "$pid" "$model" "$PWD" <<'PY' || return 1
import os, sys
from pathlib import Path
proc = Path('/proc') / sys.argv[1]
args = (proc / 'cmdline').read_bytes().decode().split('\0')
def option(name):
    return args[args.index(name) + 1]
assert proc.stat().st_uid == os.getuid() and (proc / 'cwd').resolve() == Path(sys.argv[3])
assert 'scripts/experiments/serve_low_duty.py' in args
assert (int(option('--devices')), int(option('--port')), float(option('--duty')), int(option('--reserve-mib'))) == (2, 18773, .28, 4608)
assert Path(option('--model')).resolve() == Path(sys.argv[2]).resolve()
PY
    for gpu in 5 6; do
        nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | grep -Fxq "$pid" || return 1
    done
}
free_mib() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits; }
case ${1:-run} in
    check-writer) check_writer; exit ;;
    writer)
        for gpu in 5 6; do (( $(free_mib "$gpu") >= 13824 )) || exit 75; done
        exec "$MEMSKILL_PYTHON" -u scripts/experiments/serve_low_duty.py --devices 2 --reserve-mib 4608 --duty .28 --port 18773 --model "$model" ;;
    client)
        (( $(free_mib 4) >= 4352 )) || exit 75
        check_writer || exit 75
        exec bash scripts/experiments/paired_grounding.sh training-witness-four-paths --study-forecast "$out/registration.json" --log-dir "$out/encoder-logs" ;;
    run) ;;
    *) exit 2 ;;
esac
"${program[@]}"
server=''
cleanup() {
    if [[ -n $server ]]; then
        kill -TERM "$server" 2>/dev/null || true
        wait "$server" || true
        server=''
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
until [[ -s "$out/witness/summary.json" ]]; do
    action=$(python3 - "$out" "$base" <<'PY'
import json, sys
from pathlib import Path
out, base = map(Path, sys.argv[1:])
trial = out.parent
starts = json.loads((out / 'registration.json').read_bytes())['starts']
names = [f"{r['dialogue']}_s{r['start']}" for r in starts]
eligible = [n for n in names if (trial/'witness/forecast.json').exists() or (base/'late-pairs/outcomes'/n/'pair.json').exists()]
complete = {n for n in names if (out/'cells'/n/'complete.json').exists()}
stale = any((trial/stage/'forecast.json').exists() and not (out/stage/f'through-{len(complete):02d}.json').exists() for stage in ('group', 'witness'))
print('score' if complete and stale else 'measure' if any(n not in complete for n in eligible) else 'wait')
PY
)
    if [[ $action == score ]]; then
        python3 "$guard" run --gpus 4 -- "${program[@]}" --score >>"$out/client.log" 2>&1
        continue
    fi
    if [[ $action == wait ]]; then cleanup; sleep 45; continue; fi
    if ! check_writer; then
        if [[ -n $server ]] && ! kill -0 "$server" 2>/dev/null; then
            rc=0; wait "$server" || rc=$?; server=''
            if (( rc != 0 && rc != 75 )); then exit "$rc"; fi
        fi
        if [[ -z $server && -z $(ss -H -ltn 'sport = :18773') ]] && ! systemctl --user is-active --quiet memskill-251-linked-eight; then
            python3 "$guard" run --gpus 5,6 -- bash "$0" writer >>"$out/server.log" 2>&1 &
            server=$!
        fi
        sleep 25; continue
    fi
    rc=0
    python3 "$guard" run --gpus 4 -- bash "$0" client >>"$out/client.log" 2>&1 || rc=$?
    if (( rc != 0 && rc != 75 )); then exit "$rc"; fi
    sleep 45
done
