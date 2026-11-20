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
validation_options=()
if "$python_bin" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1])).get("equal_sources", False))' "$STUDY_DIR/protocol.json"; then
    arms+=(plain)
fi
if [[ -n ${MEMSKILL_HOLDOUT_REGISTRATION:-} ]]; then
    validation_options+=(--stdout-only)
    "$python_bin" - "$STUDY_DIR" "$MEMSKILL_HOLDOUT_REGISTRATION" "${arms[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

study, manifest = map(Path, sys.argv[1:3])
raw = manifest.read_bytes()
spec = json.loads(raw)
cohort = Path("experiments/controller/protocols/paired_grounding_cohort.json")
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
if (os.environ.get("MEMSKILL_BACKEND_ID") != spec["backend"] or spec["arms"] != sys.argv[3:]
        or sha(cohort) != spec["cohort_sha256"]
        or any(sha(study / name) != value for name, value in spec["inputs"].items())):
    raise ValueError("Held-out run differs from the registered inputs, backend or controls")
dest = study / "holdout/registration.json"
if dest.exists():
    if dest.read_bytes() != raw:
        raise ValueError("This assessment already has a different registration")
else:
    if any((study / phase).exists() for phase in ("test", "longmemeval")):
        raise ValueError("Register a prospective assessment before opening its test cohorts")
    with TemporaryDirectory(dir=dest.parent) as tmp:
        receipt = Path(tmp) / dest.name
        receipt.write_bytes(raw)
        os.link(receipt, dest)
PY
fi
"$python_bin" -m experiments.controller.audit_paired "$STUDY_DIR" "$LOCOMO_DATA" validation --select-for-test >"$report_dir/selection.log"
bash "$runner" validation --study-arm count >"$report_dir/validation-count.log" 2>&1
"$python_bin" -m experiments.controller.audit_paired "$STUDY_DIR" "$LOCOMO_DATA" validation --arms "${arms[@]}" "${validation_options[@]}" >"$report_dir/validation.log"
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
