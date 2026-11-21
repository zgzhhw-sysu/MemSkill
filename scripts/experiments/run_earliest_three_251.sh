#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export STUDY_DIR="$work/study-linked-eight-gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MEMSKILL_BACKEND_ID=251-exchange-llama3-bf16-gpu56-duty28-qwen3-b2-duty30
export MEMSKILL_API_BASE=http://127.0.0.1:18773/v1 MEMSKILL_API_TIMEOUT_S=600
export MEMSKILL_ENCODER_DUTY=.30 MEMSKILL_ENCODER_MAX_BATCH=2
base="$work/runs/paired-251-linked-eight"
out="$base/source-compression/earliest-three-skills"
earlier="$base/source-compression/earlier-terminal-returns"
phase=${1:?early third-skill phase}
mkdir -p "$out"
export MEMSKILL_USAGE_LOG="$out/usage.jsonl"
if [[ $phase == forecast ]]; then
    while [[ ! -f "$out/forecast.json" ]]; do
        rc=0
        python3 "$guard" run --gpus 3 -- bash scripts/experiments/paired_grounding.sh training-earliest-three-skills --study-forecast "$base/late-pairs/semantic-writer/three-writer/skill-graph/complete/forecast.json" --log-dir "$out/encoder-logs" >>"$out/forecast.log" 2>&1 || rc=$?
        if ((rc!=0 && rc!=75)); then exit "$rc"; fi
        [[ -f "$out/forecast.json" ]] || sleep 25
    done
    exit 0
fi
if [[ $phase == evaluation || $phase == state-evaluation ]]; then
    policy="$out"
    flags=()
    if [[ $phase == state-evaluation ]]; then policy="$out/checkpoint-odds"; flags+=(--state); fi
    prior_state=
    while [[ ! -f "$policy/evaluation/summary.json" ]]; do
        if [[ -f "$policy/forecast.json" ]]; then
            state=$(find "$earlier/outcomes" "$out/readers" "$out/pairs" "$out/judged" -maxdepth 1 -name 'conv-*.json' -printf '%p\n' 2>/dev/null | sort || true)
            if [[ -n $state && $prior_state != "$state" || ( -f "$out/judged-complete.json" && -f "$earlier/outcomes-complete.json" ) ]]; then
                rc=0
                python3 "$guard" run --gpus 3 -- "$MEMSKILL_PYTHON" -u -m experiments.controller.earliest_skill_evaluation "$base" "$STUDY_DIR" "${flags[@]}" >>"$policy/evaluation.log" 2>&1 || rc=$?
                if ((rc==0)); then prior_state=$state; elif ((rc!=75)); then exit "$rc"; fi
            fi
        fi
        [[ -f "$policy/evaluation/summary.json" ]] || sleep 25
    done
    exit 0
fi
case $phase in
    writers) parent="$earlier/prepared" ;;
    prepared) parent="$out/writers" ;;
    readers) parent="$out/prepared" ;;
    pairs) parent="$out/readers" ;;
    repairs) parent="$out/pairs" ;;
    judged) parent="$out/repairs" ;;
    *) exit 2 ;;
esac
cmd=("$MEMSKILL_PYTHON" -u -m experiments.controller.earliest_three_skills "$base" "$STUDY_DIR" "$LOCOMO_DATA" "$phase")
until [[ -f "$out/$phase-complete.json" ]]; do
    pending=false
    if [[ -f "$out/forecast.json" ]]; then
        for f in "$parent"/conv-*.json; do
            if [[ -f $f && ! -f "$out/$phase/${f##*/}" ]]; then
                if [[ $phase == pairs && ! -f "$earlier/outcomes/${f##*/}" && $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$f") == answered ]]; then continue; fi
                pending=true; break
            fi
        done
        if [[ -d "$out/$phase" && $(find "$out/$phase" -maxdepth 1 -name 'conv-*.json' | wc -l) == 12 ]]; then pending=true; fi
    fi
    if [[ $pending == false ]]; then sleep 25; continue; fi
    rc=0
    case $phase in
        prepared|repairs) python3 "$guard" run --gpus 3 -- "${cmd[@]}" >>"$out/$phase.log" 2>&1 || rc=$? ;;
        pairs) "${cmd[@]}" >>"$out/$phase.log" 2>&1 || rc=$? ;;
        *) if bash scripts/experiments/run_witness_four_paths_251.sh check-writer; then "${cmd[@]}" >>"$out/$phase.log" 2>&1 || rc=$?; else rc=75; fi ;;
    esac
    if ((rc!=0 && rc!=75)); then exit "$rc"; fi
    if ((rc==75)); then sleep 25; fi
done
