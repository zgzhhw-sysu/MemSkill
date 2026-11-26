"""Matched on-policy update: same native transitions, frozen skills, three reward treatments."""
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from experiments.controller import file_hash, read, register
from experiments.controller.paper_f1 import PROTOCOL, load_split_cohort, question_scores
from experiments.designer.designer_pilot import prefix, save
from src.cli import set_seed
from src.data_processing.datasets import load_dataset, split_data
from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.controller import PPOBuffer
from src.episode_trace import skill_spec
from src.revision_validation import evaluate_bank
from src.trainer import get_trainer


@torch.no_grad()
def screen_forward_rank(native, checkpoint, out, history, *, skill_projection=False):
    """Test which retained ranks change only the skill at the deployed action boundary."""
    import time
    from experiments.designer.paper_date_pair import CHECKPOINT_SHA, MODEL, history_spans
    from src.controller import PPOController

    paper = read(PROTOCOL)
    rule = paper["forward_rank_boundary"]
    if skill_projection:
        rule = rule | paper["forward_rank_skill_projection"]
    saved = out / (history + ".json")
    if saved.exists() and read(saved)["method"] != rule:
        raise ValueError("Use a separate output directory for each rank study")
    cell = paper["training_boundary_swap"]["histories"][history]
    plan = read(native / "locomo/protocol.json")
    if (file_hash(checkpoint) != CHECKPOINT_SHA or plan["checkpoint"] != CHECKPOINT_SHA or plan["split"] != "train"
            or plan["model"]["model"] != MODEL or plan["encode_batch_size"] != 64
            or plan["question_keys"] != paper["datasets"]["locomo"]["development_splits"]["train"]["scored_key_sha256"]):
        raise ValueError("Expected the original LoCoMo training Controller and native cohort")
    folder = native / "locomo/native" / history
    path = folder / "controller-inputs.npz"
    if file_hash(path) != read(folder / "audit.json")["policy_inputs_sha256"]:
        raise ValueError("Native Controller inputs differ from the recorded run")
    spans = history_spans(folder)
    with np.load(path, allow_pickle=False) as archive:
        states = torch.from_numpy(archive["states"]).cuda()
        candidates = torch.from_numpy(archive["candidate_vectors"]).cuda()
        names = archive["candidate_names"].tolist()
        if archive["span_ids"].tolist() != [s["span_id"] for s in spans] or len(states) != cell["total_spans"]:
            raise ValueError("Incomplete native span panel")
    k = plan["settings"]["action_top_k"]
    torch.set_float32_matmul_precision("highest")
    policy = PPOController(state_dim=states.shape[-1], op_dim=candidates.shape[-1], hidden_dim=256,
                           device="cuda", action_top_k=k).cuda().eval()
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["controller_state_dict"])
    layer = policy.get_parameter(rule["layer"])
    original = layer.clone()
    expected = []
    for state, span in zip(states, spans, strict=True):
        info = {}
        chosen, _, _ = policy(state, candidates, deterministic=True, policy_trace=info)
        if (info["logits"] != span["policy"]["logits"] or info["first_draw_probabilities"] != span["policy"]["first_draw_probabilities"]
                or [names[j] for j in chosen] != [s["name"] for s in span["selected_skills"]]):
            raise ValueError("Native serial Controller forward differs from the original trace")
        next_skill = torch.tensor(info["logits"], device="cuda").topk(k + 1).indices[-1].item()
        expected.append((chosen, chosen[:-1] + [next_skill], np.array(info["first_draw_probabilities"])))
    duty = float(os.environ.get("MEMSKILL_ENCODER_DUTY", "0.8"))
    if not 0 < duty <= 1:
        raise ValueError("Invalid GPU duty")
    def measure(weight, keep_rows=False):
        torch.cuda.synchronize()
        started = time.monotonic()
        layer.copy_(weight)
        counts = Counter()
        maximum = 0.0
        rows = []
        for state, (native_order, swap_order, probabilities) in zip(states, expected, strict=True):
            info = {}
            chosen, _, _ = policy(state, candidates, deterministic=True, policy_trace=info)
            shift = float(np.max(np.abs(np.asarray(info["first_draw_probabilities"]) - probabilities)))
            kind = "native" if chosen == native_order else "boundary" if chosen == swap_order else "other"
            counts[kind] += 1
            counts["membership"] += set(chosen) != set(native_order)
            counts["probability"] += shift > 0
            maximum = max(maximum, shift)
            if keep_rows:
                rows.append({"span": len(rows), "class": kind, "ordered_skills": [names[j] for j in chosen],
                             "first_draw_probabilities": info["first_draw_probabilities"]})
        torch.cuda.synchronize()
        time.sleep((time.monotonic() - started) * (1 / duty - 1))
        return {key: counts[key] for key in ("native", "boundary", "other", "membership", "probability")} | {"max_probability_shift": maximum}, rows

    out.mkdir(parents=True, exist_ok=True)
    weights = out / "frozen-weight.json"
    result = {"history": history, "spans": len(states), "method": rule, "checkpoint_sha256": CHECKPOINT_SHA,
              "input_sha256": file_hash(path), "candidate_order": names, "device": torch.cuda.get_device_name(),
              "split": "train", "writer_reader_calls": 0, "question_labels_used": 0}
    if history == rule["calibration"]:
        u, singular, vh = torch.linalg.svd(original.double(), full_matrices=False)
        full = ((u * singular) @ vh).float()
        if not torch.equal(full, original):
            raise ValueError("Full-rank reconstruction differs from the native FP32 layer")
        selected = None
        sweep = {}
        for rank in range(len(singular) - 1, 0, -1):
            weight = ((u[:, :rank] * singular[:rank]) @ vh[:rank]).float()
            trial, _ = measure(weight)
            sweep[str(rank)] = trial
            if selected is None and trial["native"] and trial["boundary"] and not trial["other"]:
                selected = rank
                register(weights, weight.cpu().tolist())
        result["rank_sweep"] = sweep
    else:
        calibration = out / (rule["calibration"] + ".json")
        frozen = read(calibration)
        if frozen["method"] != rule or frozen["checkpoint_sha256"] != CHECKPOINT_SHA or frozen["candidate_order"] != names:
            raise ValueError("Frozen rank calibration differs from the transfer protocol")
        selected = frozen["selected_rank"]
        result["calibration_sha256"] = file_hash(calibration)
        if selected is not None and frozen["weight_sha256"] != file_hash(weights):
            raise ValueError("Frozen rank weights changed")
    result["selected_rank"] = selected
    if selected is not None:
        result["weight_sha256"] = file_hash(weights)
        result["forward"], result["rows"] = measure(torch.tensor(read(weights), device="cuda", dtype=layer.dtype), True)
        if history == rule["calibration"] and result["forward"] != result["rank_sweep"][str(selected)]:
            raise ValueError("Frozen rank did not reproduce its calibration decisions")
    register(out / (history + ".json"), result)


def load_frozen_projection(base, root, plan, paper, split):
    """Load the frozen skill matrix against the native run and original scoring receipts."""
    from experiments.designer.paper_date_pair import CHECKPOINT_SHA
    from experiments.controller.paper_f1 import read_locked_json

    training = Path(str(base) + "-locomo-train") / "locomo"
    if split == "test":
        native_reference = Path(str(base) + "-locomo-batch64") / "locomo"
        receipt = read(native_reference / "upstream-score.json")
        original = read_locked_json(native_reference / "baseline.json", receipt["baseline_sha256"])
        count = paper["datasets"]["locomo"]["scored_questions"]
        if (receipt["scorer_commit"] != paper["upstream_revision"] or receipt["questions"] != count
                or abs(receipt["f1_percent"] - original["native_f1"]) > 1e-9):
            raise ValueError("The fixed LoCoMo reference must match the author's score receipt")
        outcome = paper["frozen_skill_projection_test"]
    elif split == "train":
        native_reference = training
        outcome = paper["frozen_skill_projection_outcome"]
        verified = read(base / "paired-baselines-verified.json")
        if verified["scorer_commit"] != paper["upstream_revision"]:
            raise ValueError("Expected the author-scored paired baselines")
        for dataset, source in (("locomo", Path(str(base) + "-locomo-batch64")), ("longmemeval", base)):
            receipt = verified["datasets"][dataset]
            original = read_locked_json(source / dataset / "baseline.json", receipt["baseline_sha256"])
            if original["questions"] != receipt["questions"] or original["questions"] != paper["datasets"][dataset]["scored_questions"]:
                raise ValueError("Incomplete paired baselines")
    else:
        raise ValueError("Unsupported projection evaluation split")
    if (native_reference.resolve() == root.resolve()
            or read(native_reference / "baseline.json")["questions"] != outcome["questions"]
            or read(native_reference / "protocol.json") != plan):
        raise ValueError("Expected a separate complete native run with matching inference settings")
    study = base / "legacy-rank-audit/training-controller-dispersion/rank-skill-projection"
    rule = paper["forward_rank_boundary"] | paper["forward_rank_skill_projection"]
    sid = rule["calibration"]
    receipt = read(study / f"{sid}-upstream-verification.json")
    calibration = read_locked_json(study / f"{sid}.json", receipt["report_sha256"])
    rank = calibration["selected_rank"]
    if (calibration["method"] != rule or calibration["history"] != sid or calibration["checkpoint_sha256"] != CHECKPOINT_SHA
            or receipt["upstream"] != paper["upstream_revision"]
            or calibration["input_sha256"] != file_hash(training / "native" / sid / "controller-inputs.npz")
            or rank is None or rank != receipt["selected_rank"] or calibration["weight_sha256"] != receipt["weight_sha256"]):
        raise ValueError("Missing a verified training-selected skill projection")
    if split == "test" and (rank != outcome["rank"] or calibration["weight_sha256"] != outcome["weight_sha256"]):
        raise ValueError("Test requires the exact training-frozen projection registered for evaluation")
    projection = read_locked_json(study / "frozen-weight.json", calibration["weight_sha256"])
    metadata = {"layer": rule["layer"], "rank": rank, "weight_sha256": calibration["weight_sha256"],
                "calibration_sha256": receipt["report_sha256"], "outcome": outcome}
    return native_reference, metadata, projection


@torch.no_grad()
def projection_exposure(trainer, native_policy, directory, native_directory, frozen):
    """Compare the original policy on the exact states reached by the deployed projection."""
    from experiments.designer.paper_date_pair import history_spans
    from src.executor import deployment_operations

    src = directory / "controller-inputs.npz"
    ops = trainer.operation_bank.get_candidate_operations()
    names = [op.name for op in ops]
    with np.load(src, allow_pickle=False) as archive:
        if names != archive["candidate_names"].tolist():
            raise ValueError("The archived skill order differs from the checkpoint")
        archived_states = archive["states"]
        hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in archived_states]
        states = torch.from_numpy(archived_states).to(trainer.device)
        vectors = torch.from_numpy(archive["candidate_vectors"]).to(trainer.device)
        ids = archive["span_ids"].tolist()
    actual, original = history_spans(directory), history_spans(native_directory)
    if (not states.is_cuda or ids != [e["span_id"] for e in actual] or ids != [e["span_id"] for e in original]
            or hashes != [e["policy"]["state_sha256"] for e in actual]):
        raise ValueError("Expected complete matched native and projected GPU trajectories")
    operations = {op.name: op for op in ops}
    def prompt(order, event):
        skills = deployment_operations([operations[name] for name in order])
        text = trainer.executor._build_executor_prompt(skills, event["source"], event["retrieved"])
        return hashlib.sha256(text.encode()).hexdigest()

    rows = []
    first = None
    for j, (state, observed, reference) in enumerate(zip(states, actual, original, strict=True)):
        policy = {}
        indices, _, _ = native_policy(state, vectors, deterministic=True, policy_trace=policy)
        baseline = [names[i] for i in indices]
        selected = [s["name"] for s in observed["selected_skills"]]
        same_state = observed["policy"]["state_sha256"] == reference["policy"]["state_sha256"]
        if same_state and (policy["logits"] != reference["policy"]["logits"] or
                           baseline != [s["name"] for s in reference["selected_skills"]]):
            raise ValueError("The original policy does not reproduce the shared native state")
        observed_prompt = prompt(selected, observed)
        native_prompt = observed_prompt if baseline == selected else prompt(baseline, observed)
        if [r["prompt_sha256"] for r in observed["executor_responses"]] != [observed_prompt]:
            raise ValueError("The projected selection does not reproduce the actual writer request")
        if first is None:
            if not same_state or [r["prompt_sha256"] for r in reference["executor_responses"]] != [native_prompt]:
                raise ValueError("The native and projected runs diverged before their first different writer prompt")
            if native_prompt != observed_prompt:
                first = ids[j]
            elif observed["changes"] != reference["changes"]:
                raise ValueError("Identical prefix writer inputs produced different memory changes")
        probabilities = policy["first_draw_probabilities"]
        changed = probabilities != observed["policy"]["first_draw_probabilities"]
        members = set(selected) != set(baseline)
        rows.append({"span": ids[j], "native_trajectory_state": same_state, "native_order": baseline, "projected_order": selected,
                     "native_probabilities": probabilities, "probability": changed, "membership": members, "joint": changed and members,
                     "native_prompt": native_prompt, "projected_prompt": observed_prompt})
    counts = {key: sum(r[key] for r in rows) for key in ("probability", "membership", "joint", "native_trajectory_state")}
    counts["writer_prompt"] = sum(r["native_prompt"] != r["projected_prompt"] for r in rows)
    trace, = (directory / "traces").glob("*.jsonl")
    reference_trace, = (native_directory / "traces").glob("*.jsonl")
    return {"trace_sha256": file_hash(trace), "native_trace_sha256": file_hash(reference_trace),
            "weight_sha256": frozen["weight_sha256"], "inputs_sha256": file_hash(src), "spans": len(rows),
            "first_different_writer": first, "counts": counts, "rows": rows}


def state_features(states):
    """Give the native session and retrieved-memory embeddings equal feature weight."""
    if states.shape[-1] % 2 or not states.isfinite().all():
        raise ValueError("Expected finite concatenated session and memory vectors")
    return torch.cat([torch.nn.functional.normalize(x, dim=-1) for x in states.double().chunk(2, dim=-1)], dim=-1)


def credit_targets(measurement, linked, prior):
    deltas = {int(i): v for i, v in measurement["f1_deltas"].items()}
    if len(deltas) != measurement["questions"] or not linked or not set(linked) <= deltas.keys():
        raise ValueError("Incomplete paired training outcomes")
    if any(not -1 <= v <= 1 for v in deltas.values()):
        raise ValueError("Invalid per-question F1 difference")
    terminal = mean(deltas.values())
    if abs(100 * terminal - measurement["difference_points"]) > 1e-9:
        raise ValueError("Terminal reward differs from the per-question mean")
    return [terminal, (sum(deltas[j] for j in linked) + prior * terminal) / (len(linked) + prior)]


def fit_state_credit(states, outcomes, ridge):
    """Identical intercept and dual ridge estimator for both reward treatments."""
    x = state_features(states)
    y = outcomes.double()
    if len(x) < 1 or (y.ndim != 2 or y.shape[0] != len(x) or y.shape[1] < 2) or not y.isfinite().all() or ridge <= 0:
        raise ValueError("Expected paired finite rewards for separate training histories")
    center, intercept = x.mean(0), y.mean(0)
    z = x - center
    gram = z @ z.T + ridge * torch.eye(len(z), dtype=z.dtype, device=z.device)
    coef = z.T @ torch.linalg.solve(gram, y - intercept)
    return coef, intercept - center @ coef


def shift_boundary_logits(logits, outgoing, incoming, k, shift):
    """Keep both logits in their original interval and all other skills fixed."""
    edge = logits.topk(k + 1, dim=-1).indices[..., -2:]
    eligible = (edge == outgoing).any(-1) & (edge == incoming).any(-1)
    gap = logits[..., incoming] - logits[..., outgoing]
    target = (gap + shift.to(logits)).clamp(-gap.abs(), gap.abs())
    delta = torch.where(eligible, (target - gap) / 2, 0.)
    result = logits.clone()
    result[..., incoming] += delta
    result[..., outgoing] -= delta
    return result, eligible


def match_skill_rate(logits, shift, outgoing, incoming, k, limit):
    """Match the learned hard selection count on native training states with a constant shift."""
    learned, eligible = shift_boundary_logits(logits, outgoing, incoming, k, shift)
    def counts(x):
        return (x.topk(k, dim=-1).indices == incoming).any(-1).sum(-1)
    if logits.dtype != torch.float32 or limit <= 0:
        raise ValueError("Rate matching requires native FP32 logits and a positive shift limit")
    native, target = int(counts(logits)), int(counts(learned))
    direction = 1 if target >= native else -1
    last = int(np.float32(limit).view(np.uint32))
    visited = {0: native}

    def shift_at(bits):
        return direction * float(np.uint32(bits).view(np.float32))

    def visits(bits):
        if bits not in visited:
            adjusted, _ = shift_boundary_logits(logits, outgoing, incoming, k, logits.new_tensor(shift_at(bits)))
            visited[bits] = int(counts(adjusted))
        return visited[bits]

    def first_reaching(number):
        # Nonnegative finite FP32 magnitudes have the same order as their unsigned bits.
        # Selection count is monotone as the incoming skill rises and its competitor falls.
        left, right = 0, last
        while left < right:
            mid = (left + right) // 2
            if direction * (visits(mid) - number) >= 0:
                right = mid
            else:
                left = mid + 1
        return left

    endpoint = visits(last)
    reachable = min(target, endpoint) if direction > 0 else max(target, endpoint)
    upper = first_reaching(reachable)
    closest = min((upper, max(upper - 1, 0)), key=lambda i: (abs(visits(i) - target), i))
    best = first_reaching(visits(closest))
    static, _ = shift_boundary_logits(logits, outgoing, incoming, k, logits.new_tensor(shift_at(best)))
    return {"shift": shift_at(best), "spans": len(logits), "eligible": int(eligible.sum()),
            "native_selected": native, "span_selected": target, "rate_selected": int(counts(static)),
            "disagreement_with_span": int(((static.topk(k).indices == incoming).any(-1)
                                            != (learned.topk(k).indices == incoming).any(-1)).sum())}



def freeze_paper_credit(train, swaps, output, device, data_file, repair_ledger=None):
    """Use only registered training spans and both independently generated pairs."""
    if torch.device(device).type != "cuda":
        raise ValueError("Paper Controller fitting requires the GPU lease")
    paper = read(PROTOCOL)
    panel, spec = paper["training_boundary_swap"], paper["state_credit"]
    certified = read(repair_ledger) if repair_ledger else None
    if certified is not None and (certified["split"] != "train" or not certified["complete"]
            or certified["scorer_commit"] != paper["upstream_revision"] or certified["rule"] != paper["repair_credit"]
            or set(certified["histories"]) != set(panel["histories"])):
        raise ValueError("Need author-verified native writing and switch repeats for all training histories")
    protocol = read(train / "protocol.json")
    if (protocol.get("split") != "train" or protocol.get("dataset") != "locomo"
            or protocol.get("question_keys") != paper["datasets"]["locomo"]["development_splits"]["train"]["scored_key_sha256"]
            or protocol.get("encode_batch_size") != 64):
        raise ValueError("Unexpected native training protocol")
    _, questions = load_split_cohort("locomo", Path(data_file), paper["datasets"]["locomo"], "train")
    inputs, rewards, sources, included = [], [], {}, []
    history_states, history_logits = [], []
    candidate_names = None
    for sid, cell in panel["histories"].items():
        index = cell["span"]
        folder = train / "native" / sid
        archived = folder / "controller-inputs.npz"
        archive_hash = file_hash(archived)
        if archive_hash != read(folder / "audit.json")["policy_inputs_sha256"]:
            raise ValueError("Native training Controller inputs changed")
        with np.load(archived, allow_pickle=False) as bank:
            if bank["span_ids"].tolist() != list(range(cell["total_spans"])):
                raise ValueError("Incomplete native Controller states")
            states = bank["states"]
            state = states[index]
            names = bank["candidate_names"].tolist()
        trace, = (folder / "traces").glob("*.jsonl")
        with trace.open() as stream:
            policies = [row["policy"] for line in stream if (row := json.loads(line))["kind"] == "span"]
        if ((candidate_names is not None and names != candidate_names)
                or any(p["candidates"] != names or p["state_sha256"] != hashlib.sha256(x.tobytes()).hexdigest()
                       for p, x in zip(policies, states, strict=True))):
            raise ValueError("Native training trajectory differs from the archived Controller inputs")
        candidate_names = names
        history_states.append(states)
        history_logits.extend(p["logits"] for p in policies)
        state_hash = hashlib.sha256(state.tobytes()).hexdigest()
        targets, receipts, pairs = [], [], []
        subset = {key: qa for key, qa in questions.items() if key[0] == sid}
        for repeat in range(1, panel["independent_repeats"] + 1):
            root = swaps / sid / f"span-{index}" / "repeats" / str(repeat) / "locomo"
            path = root / "swap" / sid / "measurement.json"
            run_protocol = read(root / "protocol.json")
            fork = run_protocol.pop("training_swap")
            m = read(path)
            if (run_protocol != protocol or (fork["history"], fork["span"], fork["repeat"]) != (sid, index, repeat)
                    or m["sample_id"] != sid or m["repeat"] != repeat or not m["registered_span"]
                    or m["state_sha256"] != state_hash or m["span_id"] != index or m["matched_prefix_spans"] != index
                    or m["linked_question_indices"] != cell["linked_question_indices"] or m["questions"] != cell["questions"]
                    or not m["writer_prompt_changed"]):
                raise ValueError(f"Repeat differs from its native training fork at {sid}/{repeat}")
            if (m["artifacts"]["scores.json"] != file_hash(path.parent / "scores.json")
                    or m["native_scores_sha256"] != file_hash(root / "native" / sid / "scores.json")):
                raise ValueError("Paired answers changed after measurement")
            left, right = [question_scores("locomo", subset, [root / arm / sid / "scores.json"]) for arm in ("native", "swap")]
            if {str(k[1]): right[k] - left[k] for k in subset} != m["f1_deltas"]:
                raise ValueError("Published credit disagrees with the original per-question answers")
            pairs.append((m["outgoing"], m["incoming"]))
            targets.append(credit_targets(m, cell["linked_question_indices"], spec["prior_questions"]))
            receipts.append(file_hash(path))
        if len(set(pairs)) != 1:
            raise ValueError("Independent repeats disagree on the original boundary pair")
        sources[sid] = {"pair": pairs[0], "span": index, "state_sha256": state_hash, "inputs_sha256": archive_hash,
                        "trace_sha256": file_hash(trace), "measurements": receipts, "repeat_targets": targets}
        if set(pairs[0]) == {spec["outgoing"], spec["incoming"]}:
            sign = 1 if pairs[0][1] == spec["incoming"] else -1
            inputs.append(state)
            values = [mean(column) for column in zip(*targets, strict=True)]
            if certified is not None:
                row = certified["histories"][sid]
                if row["swap_receipts"] != receipts or row["questions"] != len(subset) or row["span"] != index:
                    raise ValueError("Independent repair labels refer to a different training action")
                if abs(values[0] - row["terminal"]) > 1e-12:
                    raise ValueError("Raw terminal disagrees with independent author scoring")
                values = [row[k] for k in paper["repair_credit"]["treatments"]]
                sources[sid]["discounted_questions"] = row["discounted"]
            rewards.append([sign * value for value in values])
            included.append(sid)
    if len(included) < 2:
        raise ValueError("Fewer than two training histories measured the registered skill pair")
    states = torch.as_tensor(np.stack(inputs), device=device)
    y = torch.tensor(rewards, device=device, dtype=torch.float64)
    for i, sid in enumerate(included):
        keep = torch.arange(len(y), device=device) != i
        w, b = fit_state_credit(states[keep], y[keep], spec["ridge"])
        sources[sid]["leave_history_out_prediction"] = (state_features(states[i]) @ w + b).tolist()
    w, b = fit_state_credit(states, y, spec["ridge"])
    model = {"schema": 1, "treatments": ["terminal", "span"], "spec": spec, "train_protocol_sha256": file_hash(train / "protocol.json"),
             "checkpoint": protocol["checkpoint"], "sources": sources, "weight": w.T.tolist(), "bias": b.tolist()}
    if certified is not None:
        model.update(treatments=paper["repair_credit"]["treatments"], repair_rule=paper["repair_credit"],
                     repair_ledger_sha256=file_hash(repair_ledger))
    all_states = torch.as_tensor(np.concatenate(history_states), device=device)
    all_logits = torch.tensor(history_logits, dtype=torch.float32, device=device)
    shift = spec["max_logit_shift"] * torch.tanh((state_features(all_states) @ w[:, -1] + b[-1]) / spec["reward_scale"])
    model["rate"] = match_skill_rate(all_logits, shift, candidate_names.index(spec["outgoing"]),
                                     candidate_names.index(spec["incoming"]), protocol["settings"]["action_top_k"], spec["max_logit_shift"])
    register(output, model)
    return model


@contextmanager
def learned_skill_boundary(trainer, model, arm):
    """Apply the frozen pre-write state score inside the original Controller forward."""
    idx = model["treatments"].index(arm) if arm != "rate" else None
    spec = model["spec"]
    if idx is not None:
        w = torch.tensor(model["weight"][idx], dtype=torch.float64, device=trainer.device)
        b = w.new_tensor(model["bias"][idx])
    names = [op.name for op in trainer.operation_bank.get_candidate_operations()]
    outgoing, incoming = (names.index(spec[k]) for k in ("outgoing", "incoming"))
    k = trainer.controller.action_top_k
    records = []
    state = None

    def capture(module, args):
        nonlocal state
        state = args[0]

    def adjust(module, args, output):
        logits = output[0, :, 0]
        value = None
        if arm == "rate":
            shift = logits.new_tensor(model["rate"]["shift"])
        else:
            value = state_features(state) @ w + b
            shift = spec["max_logit_shift"] * torch.tanh(value / spec["reward_scale"])
        adjusted, eligible = shift_boundary_logits(logits, outgoing, incoming, k, shift)
        result = adjusted.reshape_as(output)
        records.append({"eligible": bool(eligible), "prediction": value.item() if value is not None else None,
                        "native_logits": logits.tolist(), "changed_logits": adjusted.tolist(),
                        "native_top_k": [names[i] for i in logits.topk(k).indices.tolist()],
                        "native_probabilities": logits.softmax(0).tolist()})
        return result

    with trainer.controller.register_forward_pre_hook(capture), trainer.controller.actor_head.register_forward_hook(adjust):
        yield records


def log_margins(trainer, batch, alternatives):
    """Compare joint log-probability of two ordered K-tuples on the sampled states."""
    def joint(i, action):
        state = torch.as_tensor(batch.states[i], device=trainer.device).float()[None]
        ops = torch.as_tensor(batch.op_embs[i], device=trainer.device).float()[None]
        mask = torch.as_tensor(batch.new_op_masks[i], device=trainer.device)[None]
        action = torch.tensor(action, device=trainer.device).reshape(1, -1)
        with torch.no_grad():
            logp, *_ = trainer.controller.evaluate_actions(state, ops, action, new_op_masks=mask)
        return logp.item()
    return [joint(i, batch.actions[i]) - joint(i, alt) if alt is not None else None
            for i, alt in enumerate(alternatives)]


def run():
    args = get_agentic_memory_args()
    if not args.span_credit_weight or args.enable_designer:
        raise ValueError("Matched study requires positive span credit and a fixed skill bank")
    out = Path(args.out_file).parent
    out.mkdir(exist_ok=False, parents=True)
    train, val, _ = split_data(load_dataset(args.data_file, "locomo"), "locomo")
    fit = [prefix(train[i], 2) for i in (0, 1, 4, 5)]
    held = [prefix(c, 3) for c in val]
    def dataset_ids(group):
        return {c["sample_id"]: [q["original_index"] for q in c["qa"]] for c in group}
    spec = {"seed": args.seed, "train": dataset_ids(fit), "validation": dataset_ids(held),
            "checkpoint": hashlib.sha256(Path(args.load_checkpoint).read_bytes()).hexdigest(),
            "dataset": hashlib.sha256(Path(args.data_file).read_bytes()).hexdigest(),
            "config": {k: v for k, v in vars(args).items() if k != "api_key"},
            "arms": ["B0_upstream", "B1_terminal", "C1_credit"],
            "comparison": "one paired native PPO update from identical sampled episodes and Adam state"}
    save(out / "protocol.json", spec)
    set_seed(args.seed)
    trainer = get_trainer(args, AgenticMemoryConfig().update_from_args(args))
    trainer.load_checkpoint(args.load_checkpoint)
    set_seed(args.seed)
    bank = trainer.operation_bank
    versions = [skill_spec(op) for op in bank.get_candidate_operations()]
    names = [s["name"] for s in versions]
    spec["skills"] = versions
    save(out / "protocol.json", spec)
    batch = PPOBuffer()
    upstream, terminal, auxiliary, alternatives, samples = [], [], [], [], []
    os.environ["MEMSKILL_USAGE_LOG"] = str(out / "train_usage.jsonl")
    for c in fit:
        local = PPOBuffer()
        sessions, length, precompute = trainer._prepare_sessions(c)
        memory = trainer._initialize_memory_bank()
        log = trainer.train_episode(c, memory, sessions, local, episode_length=length, precompute_embeddings=precompute)
        records = [json.loads(s) for s in memory.trace.path.read_text().splitlines()]
        diagnostics = [r["diagnosis"] for r in records if r["kind"] == "span_credit"]
        for i, (d, row) in enumerate(zip(diagnostics, log["reward_components"], strict=True)):
            upstream.append(row["upstream_reward"])
            terminal.append(log["total_reward"] if i == len(diagnostics)-1 else 0.0)
            auxiliary.append(row["credit"])
            alternatives.append([names.index(s["name"]) for s in d["alternative"]] if d.get("alternative") else None)
            samples.append({"sample": c["sample_id"], "span_id": i, "terminal_f1": log["total_reward"],
                            "labels": dict(Counter(q["cause"] for q in d["questions"])),
                            "sha": memory.trace.path.name, **row})
        batch.merge(local)
        save(out / "rollout.json", samples)
        print("collected", c["sample_id"], "span credit", [r["credit"] for r in log["reward_components"]], flush=True)

    np.savez_compressed(out / "on_policy.npz", states=np.stack(batch.states), candidates=np.stack(batch.op_embs),
                        actions=batch.actions, old_log_probs=batch.log_probs, old_values=batch.values,
                        new_op_masks=batch.new_op_masks, dones=batch.dones, upstream=upstream,
                        credit=batch.rewards, terminal=terminal)
    initial = copy.deepcopy(trainer.controller.state_dict())
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    initial_steps = trainer.total_steps
    margins_before = log_margins(trainer, batch, alternatives)
    result = {"arms": {}, "margins_before": margins_before}
    # Reward vectors differ only in these treatments. No counterfactual transition enters the buffer.
    treatments = {"B0_upstream": upstream, "B1_terminal": terminal, "C1_credit": list(batch.rewards)}
    for name, rewards in treatments.items():
        set_seed(args.seed + 1000)
        trainer.controller.load_state_dict(initial)
        trainer.optimizer.load_state_dict(copy.deepcopy(optimizer))
        trainer.total_steps = initial_steps
        batch.rewards = list(rewards)
        raw_returns, advantages = batch.compute_returns_and_advantages(trainer.controller.gamma, trainer.controller.gae_lambda)
        update = trainer.update_controller(batch)
        margins_after = log_margins(trainer, batch, alternatives)
        os.environ["MEMSKILL_USAGE_LOG"] = str(out / (name + "_validation_usage.jsonl"))
        scores = evaluate_bank(trainer, bank, held, out / name)
        assert versions == [skill_spec(op) for op in trainer.operation_bank.get_candidate_operations()]
        torch.save(trainer.controller.state_dict(), out / name / "controller.pt")
        summaries = {c["sample_id"]: mean(r["f1"] for r in scores["answers"] if r["sample_id"] == c["sample_id"])
                     for c in held}
        result["arms"][name] = {"validation_f1": scores["f1"], "count": scores["count"], "by_conversation": summaries,
                                "update": update, "returns": raw_returns.tolist(), "advantages": advantages.tolist(),
                                "margins_after": margins_after}
        save(out / "results.json", result)
        print(name, scores["count"], scores["f1"], flush=True)



def probe_writer_order(base, target, corpus, phase):
    """Execute the previously checked rank-3/4 shadow intervention at the six frozen training pivots."""
    import re
    from collections import Counter
    from types import SimpleNamespace
    from experiments.designer.paper_date_pair import CHECKPOINT_SHA
    from experiments.controller import text_hash
    from src.eval.locomo import LoCoMoEvaluator
    from src.evaluation_cache import read_response, request_cache
    from src.executor import Executor, deployment_operations
    from src.llm_utils import get_llm_response_via_api

    spec = read(PROTOCOL)
    design = spec["ordered_writer_probe"]
    _,cohort = load_split_cohort("locomo",corpus,spec["datasets"]["locomo"],"train")
    train = base.with_name(base.name+"-locomo-train")
    model = read(train/"locomo/protocol.json")["model"]["model"]
    if model != "meta-llama/Llama-3.1-8B-Instruct" or read(Path(os.environ["MEMSKILL_MODEL_MANIFEST"]))["model"]!=model:
        raise ValueError("Unexpected writer checkpoint")
    executor = Executor(SimpleNamespace(retriever="qwen3-embedding-0.6b"))
    ev = LoCoMoEvaluator(SimpleNamespace())
    api = os.environ["MEMSKILL_API_BASE"]
    def tokens(text):
        return re.findall("[a-z0-9]+",text.lower())
    plans, work = {}, {}
    target.mkdir(exist_ok=True,parents=True)
    for sid in design["histories"]:
        h = train/"locomo/native"/sid
        trace, = (h/"traces").glob("*.jsonl")
        pivot = spec["training_boundary_swap"]["histories"][sid]["span"]
        with trace.open() as stream:
            span = next(e for l in stream if (e:=json.loads(l))["kind"]=="span" and e["span_id"]==pivot)
        shadow = base/"legacy-rank-audit/training-controller-dispersion"/("writer-order-"+sid+".json")
        verified = read(shadow); row, = (r for r in verified["rows"] if r["span"]==pivot)
        if verified["states_sha256"]!=file_hash(h/"controller-inputs.npz") or verified["checkpoint_sha256"]!=CHECKPOINT_SHA:
            raise ValueError("Wrong CUDA shadow selection")
        skills = [SimpleNamespace(**r) for r in span["selected_skills"]]
        if [s.name for s in skills[2:4]] != ["capture_temporal_context","capture_activity_details"]:
            raise ValueError("Expected the two frozen skill positions")
        other = skills.copy(); other[2],other[3] = other[3],other[2]
        prompts = {a:executor._build_executor_prompt(deployment_operations(v),span["source"],span["retrieved"])
                   for a,v in (("native",skills),("reordered",other))}
        hashes={a:text_hash(p) for a,p in prompts.items()}
        receipt,=span["executor_responses"]
        if (hashes["native"]!=receipt["prompt_sha256"] or hashes["native"]==hashes["reordered"]
                or any(hashes[a]!=row[a+"_prompt_sha256"] or row[a+"_tokens"]!=receipt["input_tokens"] for a in hashes)):
            raise ValueError("Writer prompts or token counts differ from the real controller shadow")
        words=" "+" ".join(tokens(span["source"]))+" "
        linked = {str(i):" ".join(tokens(ev.get_ground_truth(q))) for (h,i),q in cohort.items()
                  if h==sid and set(q.get("evidence",[]))&set(span["source_turns"])}
        linked = {i:answer for i,answer in linked.items() if len(answer)>=4 and " "+answer+" " in words}
        plans[sid]={"span":pivot,"prestate":span["policy"]["state_sha256"],"shadow_sha":file_hash(shadow),
             "trace_sha":file_hash(trace),"hashes":hashes,"input_tokens":row["native_tokens"],"references":linked}
        work[sid]=(prompts,len(span["retrieved"]))
    register(target/"plan.json",{"design":design,"model":model,"histories":plans})
    if phase=="plan":return
    for rep in (0,1):
        cell=target/str(rep);cell.mkdir(exist_ok=True)
        os.environ["MEMSKILL_GREEDY_EVAL_CACHE"]=str(cell/"responses")
        os.environ["MEMSKILL_USAGE_LOG"]=str(cell/"usage.jsonl")
        for number,sid in enumerate(design["histories"]):
            path=cell/(sid+".json")
            if phase=="measure" and path.is_file(): continue
            prompts,count=work[sid];results={}
            order=("native","reordered") if (number+rep)%2==0 else ("reordered","native")
            for a in order:
                p=prompts[a]
                if phase=="measure":
                    get_llm_response_via_api(prompt=p,LLM_MODEL=model,base_url=api,api_key="local-only",MAX_TOKENS=2048,
                                             TAU=0.0,MAX_TRIALS=10,TIME_GAP=3,usage_kind="executor")
                payload={"model":model,"messages":[{"role":"user","content":p}],"temperature":0.0,"top_p":1.0,"seed":42,"max_tokens":2048}
                _,key=request_cache(api,payload,"executor")
                hit=read_response(cell/"responses",key)
                if not hit or hit["payload"]["prompt_sha256"]!=plans[sid]["hashes"][a] or hit["payload"]["tokens"][0]!=plans[sid]["input_tokens"]:
                    raise ValueError("Writer did not return the frozen same-length prompt")
                raw=hit["payload"]["response"]
                acts=executor._parse_response(raw,count)
                writes=[r.memory_content for r in acts if r.success and r.action_type in ("INSERT","UPDATE")]
                normalized=[" "+" ".join(tokens(w))+" " for w in writes]
                hits=[i for i,v in plans[sid]["references"].items() if any(" "+v+" " in w for w in normalized)]
                results[a]={"key":key,"response_sha":hit["sha256"],"raw_sha":text_hash(raw),"tokens":hit["payload"]["tokens"],
                    "actions":dict(Counter(r.action_type for r in acts if r.success)),"failed":sum(not r.success for r in acts),
                    "writes":len(writes),"distinct_texts":len(set(writes)),"literal_hits":sorted(hits)}
            register(path,results)
            print("writer-order",rep,sid,{a:(r["writes"],r["distinct_texts"],r["literal_hits"]) for a,r in results.items()},flush=True)

if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["writer-order"]:
        import argparse
        parser = argparse.ArgumentParser(description=probe_writer_order.__doc__)
        for arg in ("base","target","corpus"):
            parser.add_argument(arg,type=Path)
        parser.add_argument("--phase",choices=("plan","measure","verify"),default="plan")
        probe_writer_order(**vars(parser.parse_args(sys.argv[2:])))
    elif sys.argv[1:2] == ["rank-boundary"]:
        import argparse
        parser = argparse.ArgumentParser(description=screen_forward_rank.__doc__)
        for arg in ("native", "checkpoint", "out"):
            parser.add_argument(arg, type=Path)
        parser.add_argument("history")
        parser.add_argument("--skill-projection", action="store_true",
                            help="Screen the preregistered final skill encoder projection")
        screen_forward_rank(**vars(parser.parse_args(sys.argv[2:])))
    else:
        run()
