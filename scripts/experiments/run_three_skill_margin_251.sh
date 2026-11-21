#!/usr/bin/env bash
set -euo pipefail
source "$HOME/memskill-work/deploy/env-251.sh"
cd "$work/run-linked-questions-v1"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
base="$work/runs/paired-251-linked-eight"
parent="$base/late-pairs/semantic-writer/three-writer"
mode=${1:-margin}
gpu=4
case $mode in
    margin)
        out="$parent/margin"
        cmd=(experiments.controller.three_skill_margin "$base" "$work/study-linked-eight-gpu" "$LOCOMO_DATA") ;;
    earliest-credit-update)
        gpu=3
        out="$parent/earliest-credit-update"
        cmd=(experiments.controller.earliest_credit_update "$base")
        early="$base/source-compression/earliest-three-skills"
        until [[ -f "$early/forecast.json" ]]; do sleep 15; done
        "$MEMSKILL_PYTHON" -m "${cmd[@]}" --register-only
        until [[ -f "$early/judged-complete.json" && -f "$parent/checkpoint-odds/forecast.json" ]]; do sleep 30; done ;;
    writer-format-sensitivity|writer-format-verified|writer-format-dense|writer-format-history|writer-format-transfer|writer-format-proxy|writer-format-sample1|writer-format-sample2)
        gpu=1
        out="$parent/$mode"
        stage=${mode#writer-format-}
        cmd=(experiments.controller.writer_format_policy "$base" "$stage")
        input="$base/source-compression/writer-format-credit/registration.json"
        case $stage in
            verified|transfer) input="${input%/*}/verified-complete.json" ;;
            proxy|sample1|sample2) gpu=0; input="${input%/*}/sampling/$stage.json" ;;
            dense) input="${input%/*}/readers-complete.json" ;;
        esac
        if [[ $stage == transfer ]]; then
            earlier="$base/source-compression/earliest-three-skills"
            until [[ -f "$parent/writer-format-verified/forecast.json" && -f "$earlier/writer-format-verified/forecast.json" ]]; do sleep 20; done
        fi
        until [[ -f "$input" && -f "$parent/checkpoint-odds/forecast.json" ]]; do sleep 20; done ;;
    outlier-total|outlier-components)
        gpu=0
        out="$parent/$mode"
        cmd=(experiments.controller.shared_outlier_policy "$base" "$mode")
        early="$base/source-compression/earliest-three-skills"
        shared="$base/source-compression/writer-format-credit/shared"
        until [[ -f "$early/temporal-gate/forecast.json" && -f "$shared/registration.json" ]]; do sleep 15; done
        if [[ $mode == outlier-components ]]; then until [[ -f "$shared/outcomes-complete.json" ]]; do sleep 20; done; fi ;;
    temporal-gate)
        gpu=0
        out="$parent/temporal-gate"
        cmd=(experiments.controller.three_skill_temporal_gate "$base")
        until [[ -f "$parent/writer-format-verified/forecast.json" && -f "$base/source-compression/earliest-three-skills/writer-format-verified/forecast.json" ]]; do sleep 20; done
        "$MEMSKILL_PYTHON" -m "${cmd[@]}" --register-only ;;
    checkpoint-odds)
        gpu=3
        out="$parent/checkpoint-odds"
        cmd=(experiments.controller.checkpoint_odds_credit "$base")
        until [[ -f "$base/source-compression/earliest-three-skills/forecast.json" ]]; do sleep 15; done ;;
    direct-step)
        gpu=3
        out="$parent/direct-step"
        cmd=(experiments.controller.direct_credit_step "$base")
        until [[ -f "$parent/skill-graph/complete/forecast.json" && -f "$parent/direct-allocation/time/forecast.json" ]]; do sleep 15; done ;;
    graph-attribution)
        gpu=3
        graph="$parent/skill-graph"
        out="$graph/attribution"
        cmd=(experiments.controller.skill_graph_attribution "$base")
        until [[ -f "$graph/masked/forecast.json" ]]; do sleep 10; done ;;
    graph-masked|graph-complete|graph-expanded)
        phase=${mode#graph-}
        gpu=3
        out="$parent/skill-graph/$phase"
        cmd=(experiments.controller.skill_graph_policy "$base" "$phase")
        direct="$base/source-compression/activity-relation-repairs/0"
        until [[ -f "$parent/margin/forecast.json" && -f "$direct/prepared-complete.json" ]]; do sleep 25; done
        if [[ $phase == complete ]]; then
            until [[ -f "$direct/outcomes-complete.json" && -f "$parent/skill-graph/masked/forecast.json" ]]; do sleep 25; done
        elif [[ $phase == expanded ]]; then
            until [[ -f "$parent/skill-graph/complete/forecast.json" && -f "$parent/skill-graph/extra-donors/outcomes-complete.json" ]]; do sleep 25; done
        fi ;;
    time|both|raw-reference)
        out="$parent/direct-allocation/$mode"
        cmd=(experiments.controller.direct_skill_allocation "$base" "$mode")
        audit="$base/source-compression/activity-relation-repairs"
        until [[ -f "$parent/margin/forecast.json" ]]; do sleep 10; done
        "$MEMSKILL_PYTHON" -m "${cmd[@]}" --register-only
        stage=outcomes
        if [[ $mode == raw-reference ]]; then stage=prepared; fi
        until [[ -f "$audit/0/$stage-complete.json" ]]; do sleep 35; done
        if [[ $mode == both ]]; then
            until [[ -f "$audit/1/outcomes-complete.json" && -f "$parent/direct-allocation/time/forecast.json" ]]; do sleep 35; done
        fi ;;
    *) exit 2 ;;
esac
mkdir -p "$out"
last=-1
until [[ -f "$out/summary.json" ]]; do
    count=$(find "$parent" -maxdepth 1 -name 'conv-*.json' | wc -l)
    if [[ $mode == temporal-gate || $mode == outlier-* ]]; then
        binary="$base/source-compression/retrieval-competition/pairs"
        count+=":$(find "$binary" -maxdepth 1 -name 'conv-*.json' | wc -l)"
        count+=":$(find "$base/late-pairs/outcomes" -mindepth 2 -maxdepth 2 -name pair.json | wc -l)"
    fi
    if [[ $mode == graph-attribution ]]; then
        direct="$base/source-compression/activity-relation-repairs/0/outcomes"
        count+=":$(find "$direct" -maxdepth 1 -name 'conv-*.json' | wc -l)"
        if [[ -f "$graph/complete/forecast.json" ]]; then count+=":final"; fi
    fi
    if [[ $count != "$last" ]]; then
        rc=0
        python3 "$guard" run --gpus "$gpu" -- "$MEMSKILL_PYTHON" -u -m "${cmd[@]}" >>"$out/policy.log" 2>&1 || rc=$?
        if ((rc==0)); then
            last=$count
            if [[ $mode == raw-reference || -f "$out/unavailable.json" ]]; then exit 0; fi
        elif ((rc!=75)); then exit "$rc"; fi
    fi
    sleep 35
done
