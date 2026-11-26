"""Hold frozen skill embeddings fixed across encoders and refit on matched training interventions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read
from experiments.controller.audit_training_replays import compare, probability_placement, read_windows
from experiments.controller.paired_policy import PairedPolicy, fit_anchored_credit, fit_policy, match_training_probability
from src.controller import PPOController
from src.operation_bank import OperationBank


def audit(study, checkpoint, development):
    protocol = read(study / "protocol.json")
    digest = file_hash(checkpoint)
    if digest != protocol["checkpoint_sha256"]:
        raise ValueError("Training protocol and checkpoint differ")
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = model["controller_state_dict"]
    hidden, state_dim = params["state_net.0.weight"].shape
    ops = OperationBank.from_dict(model["operation_bank"]).get_candidate_operations()
    names = [op.name for op in ops]
    if names != [skill["name"] for skill in protocol["skills"]]:
        raise ValueError("Training and checkpoint skill ordering differ")
    for op, skill in zip(ops, protocol["skills"], strict=True):
        if any(getattr(op, k) != skill[k] for k in ("description", "instruction_template", "update_type")):
            raise ValueError("Training skill differs from the embedded checkpoint skill")
    bank = torch.tensor(np.stack([op.embedding for op in ops]), dtype=torch.float32)
    net = PPOController(state_dim, bank.shape[1], hidden, device="cpu", action_top_k=3)
    net.load_state_dict(params)
    alt = names.index("capture_temporal_context") if protocol.get("equal_sources") else None
    gate = PairedPolicy(net.eval(), names.index("capture_activity_details"), alt)
    gate.mode = "native"

    def recalculate(row):
        gate.decisions.clear()
        gate(torch.tensor(row["state"]), bank, deterministic=True)
        new, = gate.decisions
        if any(row[arm] != new[arm] for arm in ("grounded", "plain")):
            raise ValueError("Stored F1 intervention used a different ordered skill pair")
        return new["base_logit"]

    windows = read_windows(study, protocol)
    rows = {(sid, start, row["span"]): dict(row, base_logit=recalculate(row))
            for (sid, start), pairs in windows.items() for row, _ in pairs}
    records = compare(study)["records"]
    for held in protocol["train"]:
        train = [row for (sid, _, _), row in rows.items() if sid != held]
        terminal = fit_policy(train, 0)
        credit = fit_anchored_credit(train, terminal)
        weights = {"canonical_terminal": terminal, "canonical_credit": credit,
                   "canonical_rate": match_training_probability(train, terminal, credit)}
        for link in records:
            if link["dialogue"] != held:
                continue
            row = rows[held, link["window"], link["span"]]
            features = F.normalize(torch.tensor(row["state"]), dim=0)
            for arm, w in weights.items():
                link["probabilities"][arm] = float((row["base_logit"] + w[0] + features @ w[1:]).sigmoid())

    development_offsets = []
    for directory in development:
        dev = read(directory / "protocol.json")
        if (dev["checkpoint_sha256"] != digest or dev["skills"] != protocol["skills"]
                or dev["input_sha256"] != protocol["input_sha256"] or dev["validation"] != protocol["validation"]):
            raise ValueError("Development protocol differs from the training skill library or data")
        by_sample = {}
        for sid in dev["validation"]:
            path = directory / "validation/credit" / sid
            decisions = read(path / "decisions.json")
            if len(decisions) != dev["validation_sessions"]:
                raise ValueError("Incomplete development trajectory")
            by_sample[sid] = {"backend": read(path / "inputs.json")["backend"],
                              "offsets": [[row["base_logit"], recalculate(row)] for row in decisions]}
        development_offsets.append(by_sample)
    return {"checkpoint_sha256": digest, "skill_embeddings_sha256": hashlib.sha256(bank.numpy().tobytes()).hexdigest(),
            "training_spans_with_matching_choices": len(rows), "measured_links": len(records),
            "one_step_expectation": probability_placement(records), "development_offsets": development_offsets}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--development", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(audit(args.study, args.checkpoint, args.development), indent=2))
