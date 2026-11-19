#!/usr/bin/env bash
# Complete both fixed test cohorts for one development-selected study, resuming by history.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
: "${STUDY_DIR:?Set the directory of the selected study}"
: "${LOCOMO_DATA:?Set the path to locomo10.json}"
: "${LONGMEMEVAL_DATA:?Set the path to longmemeval_s_cleaned.json}"
python_bin=${MEMSKILL_PYTHON:-python}
report_dir="$STUDY_DIR/holdout"
mkdir -p "$report_dir"
exec 9>"$report_dir/evaluation.lock"
flock 9
exec 8<>"$STUDY_DIR/fit.lock"
flock --shared 8
runner=scripts/experiments/paired_grounding.sh
arms=(native always credit terminal rate count)
"$python_bin" -m experiments.controller.audit_paired "$STUDY_DIR" "$LOCOMO_DATA" validation --select-for-test >"$report_dir/selection.log"
bash "$runner" validation --study-arm count >"$report_dir/validation-count.log" 2>&1
"$python_bin" -m experiments.controller.audit_paired "$STUDY_DIR" "$LOCOMO_DATA" validation --arms "${arms[@]}" >"$report_dir/validation.log"
(
    cd "$STUDY_DIR"
    if [[ -f holdout/frozen.sha256 ]]; then
        sha256sum -c holdout/frozen.sha256
    else
        sha256sum protocol.json terminal.json credit.json rate.json >holdout/frozen.sha256.tmp
        mv holdout/frozen.sha256.tmp holdout/frozen.sha256
    fi
)
for phase in test transfer; do
    dataset=$LOCOMO_DATA
    audit_phase=$phase
    if [[ $phase == transfer ]]; then
        dataset=$LONGMEMEVAL_DATA
        audit_phase=longmemeval
        bash "$runner" transfer --study-arm all >"$report_dir/transfer-all.log" 2>&1
    else
        for arm in "${arms[@]}"; do
            bash "$runner" "$phase" --study-arm "$arm" >"$report_dir/$phase-$arm.log" 2>&1
        done
    fi
    "$python_bin" -m experiments.controller.audit_paired "$STUDY_DIR" "$dataset" "$audit_phase" --arms "${arms[@]}" >"$report_dir/$phase.log"
done
(cd "$STUDY_DIR" && sha256sum -c holdout/frozen.sha256)
