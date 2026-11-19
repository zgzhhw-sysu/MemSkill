#!/usr/bin/env bash
# Replay the fixed external validation cohort for the already screened LoCoMo policy.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
: "${STUDY_DIR:?}" "${LOCOMO_DATA:?}" "${LONGMEMEVAL_DATA:?}" "${MEMSKILL_BACKEND_ID:?}"
python_bin=${MEMSKILL_PYTHON:-python}
phase=longmemeval-validation
output="$STUDY_DIR/$phase"
mkdir -p "$output"
exec 9>"$output/evaluation.lock"
flock 9
exec 8<>"$STUDY_DIR/fit.lock"
flock --shared 8
"$python_bin" -m experiments.controller.audit_paired "$STUDY_DIR" "$LOCOMO_DATA" validation --select-for-test
"$python_bin" - "$STUDY_DIR" "$phase" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

study = Path(sys.argv[1])
manifest = Path("experiments/controller/protocols/paired_external_validation.json")
fit = {p: hashlib.sha256((study / p).read_bytes()).hexdigest()
       for p in ("protocol.json", "terminal.json", "credit.json", "rate.json")}
protocol = json.loads((study / "protocol.json").read_text())
backend = os.environ["MEMSKILL_BACKEND_ID"]
for sid in protocol["validation"]:
    inputs = json.loads((study / "validation" / "credit" / sid / "inputs.json").read_text())
    if inputs["backend"] != backend:
        raise ValueError("Use the encoder and generator backend from the screened development run")
record = {"weights_and_protocol": fit, "cohort_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
          "backend": backend, "development_selection_sha256": hashlib.sha256((study / "validation/selection.json").read_bytes()).hexdigest()}
dest = study / sys.argv[2] / "frozen.json"
if dest.exists():
    if json.loads(dest.read_text()) != record:
        raise ValueError("The frozen external validation inputs have changed")
else:
    with TemporaryDirectory(dir=dest.parent) as tmp:
        path = Path(tmp) / dest.name
        path.write_text(json.dumps(record, indent=2))
        os.link(path, dest)
PY
arms=(native always credit terminal rate count)
if "$python_bin" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1])).get("equal_sources", False))' "$STUDY_DIR/protocol.json"; then
    arms+=(plain)
fi
runner=scripts/experiments/paired_grounding.sh
case ${MEMSKILL_EXTERNAL_WORKERS:-1} in
    1) bash "$runner" transfer-validation --study-arm all ;;
    2)
        workers=()
        stop_workers() {
            for pid in "${workers[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
            wait || true
        }
        trap stop_workers EXIT
        trap 'exit 143' TERM INT
        for part in even odd; do
            bash "$runner" transfer-validation --study-arm all --study-history-part "$part" --log-dir "$output/logs-$part" \
                >"$output/worker-$part.log" 2>&1 &
            workers+=("$!")
        done
        wait -n -p finished "${workers[@]}"
        for pid in "${workers[@]}"; do
            if [[ $pid != "$finished" ]]; then workers=("$pid"); wait "$pid"; fi
        done
        workers=()
        trap - EXIT TERM INT
        ;;
    *) echo "MEMSKILL_EXTERNAL_WORKERS must be 1 or 2" >&2; exit 2 ;;
esac
"$python_bin" -m experiments.controller.audit_paired "$STUDY_DIR" "$LONGMEMEVAL_DATA" "$phase" \
    --arms "${arms[@]}"
