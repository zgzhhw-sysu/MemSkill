"""Validate and restore a traced prefix, then replay both branches to final QA."""
from types import SimpleNamespace
from datetime import datetime, timedelta
import re
import hashlib
import json


def anchor_relative_time(turn):
    """Expand only directly resolvable relative dates, using the conversation timestamp."""
    header = turn.get("session_date", "").rpartition(" on ")[-1]
    try:
        date = datetime.strptime(header, "%d %B, %Y").date()
    except ValueError:
        return turn["text"]
    def anchor(match):
        phrase = match.group().lower()
        if phrase == "last year":
            value = str(date.year - 1)
        else:
            offset = {"yesterday": -1, "today": 0, "tomorrow": 1}[phrase]
            value = (date + timedelta(days=offset)).strftime("%d %B %Y").lstrip("0")
        return f"{match.group()} ({value})"
    return re.sub(r"\b(yesterday|today|tomorrow|last year)\b", anchor, turn["text"], flags=re.I)


def dated_relative_turns(source):
    """Retain complete dated dialogue turns with resolvable relative times; never use QA annotations."""
    header, _, body = source.partition("\n")
    if not header.startswith("DATE: "):
        return []
    date = header.removeprefix("DATE: ")
    quotes = []
    for line in body.splitlines():
        match = re.fullmatch(r'([^"\n]+?) said, "(.*)"(?: and shared .+\.)?', line)
        if match:
            speaker, words = match.groups()
            expanded = anchor_relative_time({"session_date": date, "text": words})
            if expanded != words:
                quotes.append(f"{date} | {speaker}: {expanded}")
    return list(dict.fromkeys(quotes))


def attributed_turn_pairs(source):
    """Keep adjacent source turns together so short replies retain speaker and date context."""
    header, _, body = source.partition("\n")
    if header.startswith("Session Date: "):
        date = header.removeprefix("Session Date: ")
        # LongMemEval full sessions contain multi-line messages with role headers.
        messages = re.split(r"(?m)^(user|assistant): ", body.strip())
        turns = []
        for role, words in zip(messages[1::2], messages[2::2], strict=True):
            turns.append(f"{role}: {words.strip()}")
        chunks = []
        for i in range(0, len(turns), 2):
            pair = " ".join(turns[i:i + 2]).split()
            # Bound individual retrieval items while preserving every source word.
            chunks.extend(f"{date} | " + " ".join(pair[j:j + 96]) for j in range(0, len(pair), 96))
        return chunks
    if not header.startswith("DATE: "):
        return []
    date = header.removeprefix("DATE: ")
    turns = []
    for line in body.splitlines():
        match = re.fullmatch(r'([^"\n]+?) said, "(.*)"(?: and shared (.+))?', line)
        if match:
            speaker, words, image = match.groups()
            words = anchor_relative_time({"session_date": date, "text": words})
            turns.append(f"{speaker}: {words}" + (f" [shared {image}]" if image else ""))
    return [date + " | " + " ".join(turns[i:i + 2]) for i in range(0, len(turns), 2)]


def restore_text(events):
    """Replay stable IDs including repeated updates and deletion of shifted indices."""
    live = {}
    versions = {}
    for event in events:
        for (mid, version), text in zip(event["refs"], event["retrieved"], strict=True):
            if versions.get(mid) != version or live.get(mid) != text:
                raise ValueError(f"Incorrect pre-write reference {mid}@{version}")
        for change in event["changes"]:
            mid, version = change["memory_id"], change["version"]
            kind = change["action"]
            if kind == "INSERT" and mid in versions:
                raise ValueError("Reused memory ID")
            if kind != "INSERT" and mid not in live:
                raise ValueError("Target memory is absent")
            if versions.get(mid, 0) + 1 != version or live.get(mid) != change["before"]:
                raise ValueError(f"Inconsistent revision {mid}@{version}")
            versions[mid] = version
            if kind == "DELETE":
                if change["after"] is not None:
                    raise ValueError("DELETE has a nonempty destination")
                del live[mid]
            elif kind in ("UPDATE", "INSERT"):
                if not isinstance(change["after"], str):
                    raise ValueError("Expected the written text")
                live[mid] = change["after"]
            else:
                raise ValueError(f"Unknown change {kind}")
    return live


def rebuild_suffix(trainer, events, start, first_actions, *, select_suffix=None, observe_suffix=None):
    """Restore the prefix and run the suffix once for any questions sharing this intervention."""
    from src.rag_utils import get_embeddings
    bank = trainer._initialize_memory_bank()
    prefix = restore_text(events[:start])
    texts = list(prefix.values())
    if texts:
        ret = get_embeddings(trainer.args.retriever, texts, "context")
        state = trainer.state_encoder._encode_texts(texts)
        for i, text in enumerate(texts):
            bank.add_memory(text, ret[i], state_encoder_embedding=state[i])
    event = events[start]
    id_to_index = {mid: index for index, mid in enumerate(prefix)}
    indices = [id_to_index[mid] for mid, _ in event["refs"]]
    actions = first_actions
    for i in range(start, len(events)):
        current = events[i]
        if i != start:
            emb = trainer.state_encoder._encode_texts(current["source"])
            if select_suffix is None:
                old, indices = bank.retrieve(emb.reshape(-1), use_state_encoder=True)
                ops = [SimpleNamespace(**s) for s in current["skills"]]
            else:
                old, indices, memory_embs = bank.retrieve(emb.reshape(-1), use_state_encoder=True, return_embeddings=True)
                ops = select_suffix(current, emb.reshape(-1), old, memory_embs)
            actions = trainer.executor.execute_operation(ops, current["source"], old)
        if not actions or not all(a.success for a in actions):
            return None, current["span_id"]
        success = trainer.executor.apply_to_memory_bank(actions, bank, indices)
        if i != start and observe_suffix is not None:
            observe_suffix(current, old, ops, actions, success)
        if not success:
            return None, current["span_id"]
        bank.step()
    return bank, None


def replay_suffix(trainer, events, start, first_actions, qa, case):
    """Hold the logged later skill choices fixed. Recompute retrieval, writes and QA."""
    from src.rag_utils import get_embeddings
    from src.llm_utils import request_task
    bank, failed_span = rebuild_suffix(trainer, events, start, first_actions)
    if bank is None:
        return {"valid": False, "failed_span": failed_span}
    emb = get_embeddings(trainer.args.retriever, [case.question], "query")[0]
    retrieved, _ = bank.retrieve(emb)
    evaluator = trainer.evaluator
    prompt = evaluator.build_prompt(case.question, retrieved, qa)
    _, answer, _, ok = request_task((0, prompt, evaluator.prepare_eval_args()))
    final_text = bank.get_all_contents()
    return {"valid": bool(ok), "answer": answer, "memories": retrieved,
            "final_text_sha256": hashlib.sha256(json.dumps(final_text).encode()).hexdigest(),
            "final_bank_matches_observed": list(restore_text(events).values()) == final_text,
            "context_matches_observed": retrieved == case.retrieved_memories,
            "f1": evaluator.compute_f1(answer, evaluator.get_ground_truth(qa), qa)}


def reader_probes(trainer, case, qa, events, turns):
    """Change one final reader slot; use existing versions and source evidence for candidate selection."""
    from src.llm_utils import request_task

    if not turns:
        return {"finding": "unresolved", "reason": "no linked source turn"}
    live = restore_text(events)
    ids = list(live)
    refs = [ids[i] for i in case.retrieved_indices]
    context = list(case.retrieved_memories)
    if context != [live[mid] for mid in refs]:
        raise ValueError("Final QA context does not match the logged versions")
    tokens = lambda text: set(re.findall(r"\w+", text.lower()))
    source = " ".join(t["text"] for t in turns)
    target = tokens(source) | tokens(case.question)

    def similarity(text):
        words = tokens(text)
        return len(target & words) / max(len(target | words), 1)

    evaluator = trainer.evaluator
    def answer(memories):
        prompt = evaluator.build_prompt(case.question, memories, qa)
        _, pred, _, ok = request_task((0, prompt, evaluator.prepare_eval_args()))
        return {"ok": bool(ok), "answer": pred,
                "f1": evaluator.compute_f1(pred, evaluator.get_ground_truth(qa), qa)}

    control = answer(context)
    out = {"control": control, "finding": "unresolved", "probes": []}
    if not control["ok"] or control["answer"].strip() != case.prediction.strip():
        out["reason"] = "reader control changed"
        return out

    candidates = []
    for event in events:
        for change in event["changes"]:
            mid = change["memory_id"]
            if change["before"] and change["before"] != live.get(mid):
                candidates.append((similarity(change["before"]), event["span_id"], change))
    # Restoring a previous version for a currently retrieved ID isolates the text change.
    visible = [(score, span, c) for score, span, c in candidates if c["memory_id"] in refs]
    if visible:
        _, span, c = max(visible, key=lambda row: row[0])
        slot = refs.index(c["memory_id"])
        edited = context.copy()
        edited[slot] = c["before"]
        out["probes"].append({"type": "prior_version", "span": span, "memory_id": c["memory_id"],
                              "version": c["version"] - 1, "reader_slot": slot, **answer(edited)})
    deleted = [(score, span, c) for score, span, c in candidates if c["action"] == "DELETE"]
    if deleted:
        _, span, c = max(deleted, key=lambda row: row[0])
        out["probes"].append({"type": "deleted_version", "span": span, "memory_id": c["memory_id"],
                              "version": c["version"] - 1, **answer([*context, c["before"]])})
    hidden = [(mid, text) for mid, text in live.items() if mid not in refs]
    if hidden:
        mid, text = max(hidden, key=lambda pair: similarity(pair[1]))
        out["probes"].append({"type": "existing_unretrieved_item", "memory_id": mid,
                              **answer([*context[:-1], text])})
    # Source ceiling can detect recoverability even for facts never written to the bank.
    out["probes"].append({"type": "source_turn", "turn_ids": [t["dia_id"] for t in turns],
                          **answer([*context[:-1], source])})
    quoted = "\n".join(f'{t.get("speaker", "")}: {t["text"]}' for t in turns)
    dated = "\n".join(f'{t["session_date"]} | {t.get("speaker", "")}: {t["text"]}' for t in turns)
    if all(t.get("session_date") for t in turns):
        for kind, item in [("speaker_source", quoted), ("dated_speaker_source", dated)]:
            out["probes"].append({"type": kind, "turn_ids": [t["dia_id"] for t in turns],
                                  **answer([*context[:-1], item])})
        anchored = "\n".join(f'{t["session_date"]} | {t.get("speaker", "")}: {anchor_relative_time(t)}' for t in turns)
        if anchored != dated:
            out["probes"].append({"type": "calendar_expansion", "turn_ids": [t["dia_id"] for t in turns],
                                  **answer([*context[:-1], anchored])})
    helped = [p["type"] for p in out["probes"] if p["ok"] and p["f1"] > control["f1"]]
    out["finding"] = "+".join(helped) + "_helped" if helped else "unresolved"
    return out
