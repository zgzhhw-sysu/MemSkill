#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
base="$work/runs/paired-251-linked-eight"
study="$work/study-linked-eight-gpu"
root="$base/source-compression/exchange-credit"
mkdir -p "$root"
export MEMSKILL_BACKEND_ID=251-exchange-llama3-bf16-gpu56-duty28-qwen3-b2-duty30
export MEMSKILL_API_BASE=http://127.0.0.1:18773/v1 MEMSKILL_API_TIMEOUT_S=600
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
role=${1:?expected questions, prepare, read, policy or backend}
writer=scripts/experiments/run_witness_four_paths_251.sh
if [[ $role == start-backend ]]; then
    [[ -z $(ss -H -ltn 'sport = :18773') ]] || exit 75
    exec bash "$writer" writer
fi
if [[ $role == backend ]]; then
    until [[ -f "$root/read.complete" && -f "$root/framed/read.complete" ]]; do
        if ! bash "$writer" check-writer && [[ -z $(ss -H -ltn 'sport = :18773') ]]; then
            # The worker rechecks the port under the same exclusive 5/6 lease as the original server.
            python3 "$guard" run --gpus 5,6 -- bash "$0" start-backend >>"$root/server.log" 2>&1 &
            owner=$!
            trap 'kill -TERM "$owner" 2>/dev/null || true; wait "$owner" || true' EXIT
            while kill -0 "$owner" 2>/dev/null && [[ ! -f "$root/read.complete" || ! -f "$root/framed/read.complete" ]]; do sleep 30; done
            kill -TERM "$owner" 2>/dev/null || true
            wait "$owner" || true
            trap - EXIT
        fi
        sleep 30
    done
    exit
fi
formats=(strict framed)
if [[ $role == questions && -n ${2:-} ]]; then formats=("$2"); fi
while true; do
    complete=0
    progressed=0
    for kind in "${formats[@]}"; do
        out="$root"
        if [[ $kind == framed ]]; then out="$root/framed"; fi
        mkdir -p "$out"
        export MEMSKILL_EXCHANGE_FORMAT="$kind" MEMSKILL_USAGE_LOG="$out/usage.jsonl"
        if [[ -f "$out/$role.complete" ]]; then ((complete+=1)); continue; fi
        if [[ $role == policy ]]; then
            [[ -f "$out/registration.json" ]] || continue
            state=$(python3 -m experiments.controller.exchange_status "$out" policy)
            [[ $state == work ]] || continue
            rc=0
            python3 "$guard" run --gpus 3 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.exchange_policy "$base" "$study" "$LOCOMO_DATA" >>"$out/policy.log" 2>&1 || rc=$?
        else
            state=$(python3 -m experiments.controller.exchange_status "$out" "$role")
            if [[ $state == done ]]; then touch "$out/$role.complete"; continue; fi
            [[ $state == work ]] || continue
            args=("$base" "$study" "$LOCOMO_DATA")
            rc=0
            if [[ $role == prepare ]]; then
                python3 "$guard" run --gpus 3 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.exchange_paths "${args[@]}" prepare >>"$out/prepare.log" 2>&1 || rc=$?
            elif bash "$writer" check-writer; then
                if [[ $role == questions ]]; then
                    "$MEMSKILL_PYTHON" -u -m experiments.controller.exchange_questions "${args[@]}" --generate 4 >>"$out/questions.log" 2>&1 || rc=$?
                else
                    "$MEMSKILL_PYTHON" -u -m experiments.controller.exchange_paths "${args[@]}" read >>"$out/read.log" 2>&1 || rc=$?
                fi
            else
                continue
            fi
        fi
        if ((rc!=0 && rc!=75)); then exit "$rc"; fi
        if ((rc==0)); then progressed=1; fi
    done
    if ((complete==${#formats[@]})); then break; fi
    if ((progressed==0)); then sleep 20; fi
done
