"""Complete the visible turn archive for the registered full-prefix endpoints."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from functools import lru_cache
from statistics import mean
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from experiments.controller import file_hash, read, register
from experiments.controller.evidence_turns import texts
from src.data_processing.locomo import LoCoMoProcessor
from src.evaluation_cache import digest, encode_batch
from src.rag_utils import init_context_model, set_retriever_flash_attn


def encode(base, study, corpus):
    group = base / "source-compression/witness-passages"
    original = group.parent / "evidence-transport/turns"
    ext = group / "evidence-advantage/prefix-sources"
    folder = ext / "turns"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = base / "late-pairs/cross-window-prefix/manifest.json"
    proto = read(study / "protocol.json")
    if (
        file_hash(corpus) != proto["input_sha256"]
        or read(manifest)["corpus"] != proto["input_sha256"]
    ):
        raise ValueError("Different full-prefix data")
    training = set(proto["train"])
    histories = {
        r["sample_id"]: r["conversation"]
        for r in read(corpus)
        if r["sample_id"] in training
    }
    targets = {(r["dialogue"], r["end"]) for r in read(manifest)["endpoints"]}
    if len(targets) != 99 or {sid for sid, n in targets} != training:
        raise ValueError("Different prefix endpoints")
    missing = sorted(
        (sid, n) for sid, n in targets if not (original / f"{sid}_s{n}.npz").exists()
    )
    lock = ext / "complete.json"
    if lock.exists():
        rec = read(lock)
        if rec["manifest"] != file_hash(manifest) or rec["files"] != {
            f"{sid}_s{n}": file_hash(folder / f"{sid}_s{n}.npz") for sid, n in missing
        }:
            raise ValueError("Changed prefix source archive")
        return
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not 0 < float(os.environ["MEMSKILL_ENCODER_DUTY"]) <= 0.30
        or int(os.environ["MEMSKILL_ENCODER_MAX_BATCH"]) != 2
    ):
        raise ValueError("Use the registered bounded CUDA encoder")
    free, total = torch.cuda.mem_get_info()
    cap = min(int(0.65 * total), free - 2048 * 2**20)
    if cap < 2048 * 2**20:
        raise SystemExit(75)
    torch.cuda.set_per_process_memory_fraction(cap / total)
    set_retriever_flash_attn(False)
    _, encoder = init_context_model(proto["retriever"])
    if not all(v.is_cuda for v in encoder.parameters()):
        raise ValueError("Model not on leased GPU")
    proc = LoCoMoProcessor(chunk_mode="full-session")

    def values(sid, n):
        hist = histories[sid]
        k = f"session_{n}"
        turns = hist[k]
        ids = [t["dia_id"] for t in turns]
        (src,) = proc._extract_session_chunks(turns, hist[k + "_date_time"])
        vec = F.normalize(
            torch.as_tensor(
                encode_batch(encoder, texts(hist, turns), 2, normalize=True),
                device="cuda",
            ).float(),
            dim=1,
        )
        return ids, digest(src), vec

    sid = min(training)
    ids, source, v = values(sid, 5)
    with np.load(original / f"{sid}_s5.npz", allow_pickle=False) as a:
        reference = torch.as_tensor(a["vectors"], device="cuda")
        diff = float((reference - v).abs().max())
        cosine = float(F.cosine_similarity(reference, v).min())
        if (
            ids != a["ids"].tolist()
            or source != str(a["source"])
            or diff > 1e-4
            or cosine < 0.999999
        ):
            raise ValueError("Encoder parity failed")
    files = {}
    for sid, n in missing:
        name = f"{sid}_s{n}"
        path = folder / (name + ".npz")
        ids, source, v = values(sid, n)
        if not path.exists():
            with TemporaryDirectory(dir=folder) as tmp:
                dst = Path(tmp) / path.name
                np.savez(dst, ids=ids, source=source, vectors=v.cpu().numpy())
                try:
                    os.link(dst, path)
                except FileExistsError:
                    pass
        with np.load(path, allow_pickle=False) as a:
            if (
                ids != a["ids"].tolist()
                or str(a["source"]) != source
                or not torch.equal(torch.as_tensor(a["vectors"], device="cuda"), v)
            ):
                raise ValueError("Changed prefix turn vectors")
        files[name] = file_hash(path)
    register(
        lock,
        {
            "manifest": file_hash(manifest),
            "corpus": file_hash(corpus),
            "files": files,
            "parity_max_error": diff,
            "parity_min_cosine": cosine,
        },
    )


def cited_contexts(context, spans, encode_ids, decode, build_prompt, rule):
    """Dereference one version-to-source pointer using only native retrieval and stored text."""
    weights, anchors, seen = {}, {}, set()
    for rank, (origin, text) in enumerate(context, 1):
        if text not in seen:
            seen.add(text)
            weights[origin] = weights.get(origin, 0.) + 1 / rank
            anchors.setdefault(origin, text)
    winner = max(weights, key=weights.get)
    foil = (winner + (len(spans) + 1) // 2) % len(spans)
    keep = rule["keep_native_ranks"]
    if len(context) != 20 or winner == foil or not 0 < keep < len(context):
        raise ValueError("Readback needs two distinct source spans and a native top 20")
    visible = [text for _, text in context]
    native = build_prompt(visible)
    limit = len(encode_ids(native))
    kept = visible[:keep]
    available = min(rule["max_excerpt_tokens"], max(0, limit - len(encode_ids(build_prompt(kept)))))
    def source(index):
        header, _, conversation = spans[index]["source"].partition("\n")
        if not header.startswith(("DATE: ", "Session Date: ")):
            raise ValueError("Expected a dated native source chunk")
        return header, encode_ids(conversation.removeprefix("CONVERSATION:\n"))
    header, original = source(winner)
    sham_header, control = source(foil)
    match = SequenceMatcher(None, original, encode_ids(anchors[winner]), autojunk=False).find_longest_match()
    center = match.a + match.size // 2 if match.size else len(original)//2

    def render(date, tokens, anchor, cap):
        size = min(cap, len(tokens))
        while size > 0:
            first = min(max(anchor - size // 2, 0), len(tokens) - size)
            excerpt = date + "\n" + decode(tokens[first:first+size]).strip()
            prompt = build_prompt([*kept, excerpt])
            excess = len(encode_ids(prompt)) - limit
            if excess <= 0:
                return prompt, excerpt, size
            size -= excess
        return build_prompt(kept), "", 0
    primary, excerpt, count = render(header, original, center, available)
    neutral, sham, _ = render(sham_header, control, center*len(control)//max(1,len(original)), count)
    return {"native": native, "source": primary, "sham": neutral}, (winner,foil), (excerpt,sham)


def readback_tokenizer(meta):
    from transformers import AutoTokenizer

    model_dir = Path(os.environ["MEMSKILL_MODEL_MANIFEST"]).parent
    if (meta["model"] != read(model_dir/"verified-model.json")
            or meta["model"]["model"] != "meta-llama/Llama-3.1-8B-Instruct"
            or meta["backend"] != os.environ["MEMSKILL_BACKEND_ID"]
            or file_hash(model_dir/"tokenizer.json") != meta["model"]["sha256"]["tokenizer.json"]):
        raise ValueError("Unexpected generation model, backend or tokenizer revision")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    @lru_cache(maxsize=2048)
    def encode_ids(text):
        return tuple(tokenizer.encode(text, add_special_tokens=False, verbose=False))
    return encode_ids, tokenizer.decode



def clue_router(spans, rule):
    """A read-time choice using only question terms and the evidence about to be displayed."""
    import math
    import re
    from collections import Counter

    stop = set(rule["stopwords"].split())
    @lru_cache(maxsize=4096)
    def terms(text):
        return frozenset(re.findall(r"[a-z0-9]{2,}",text.lower())) - stop

    df = Counter(word for span in spans for word in terms(span["source"]))
    def route(question, context, excerpt, keep):
        visible = set().union(*(terms(text) for _,text in context[:keep]))
        candidates = terms(question)-visible
        tail = set().union(*(terms(text) for _,text in context[keep:])) & candidates
        added = terms(excerpt) & candidates
        weight = lambda words: sum(1+math.log((1+len(spans))/(1+df[w])) for w in sorted(words))
        gain,loss = added-tail,tail-added
        a,b = weight(gain),weight(loss)
        return {"selected":"source" if a+1e-9>=b else "native","gained":sorted(gain),"lost":sorted(loss),
                "gain_weight":a,"loss_weight":b}
    return route



def repeat_clue_reader(directory, work, rule, paper, model, evaluator, phase, workers):
    """Freeze routing and rerun endpoints with a fresh cache, prioritizing the questions that distinguish the policies."""
    from experiments.controller import text_hash
    from experiments.controller.reader_slot_credit import measured_prompt

    previous = read(directory/"upstream.json")
    if (previous["rule"]!=rule or previous["scorer_commit"]!=paper["upstream_revision"]
            or previous["all"]["questions"]!=len(work)):
        raise ValueError("Live repetition requires an audited frozen routing panel")
    fresh = directory/"fresh"
    fresh.mkdir(exist_ok=True)
    cache = fresh/"responses"
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    os.environ["MEMSKILL_USAGE_LOG"] = str(fresh/"usage.jsonl")
    histories = {sid:read(directory/(sid+".json"))["rows"] for sid in previous["histories"]}
    frozen = {}
    for sid,i,q,prompts in work:
        frozen.setdefault(sid,{})[str(i)]={"selected":histories[sid][str(i)]["selected"],
            "hashes":{a:text_hash(prompts[a]) for a in ("native","source")}}
    register(fresh/"plan.json",{"frozen_panel":file_hash(directory/"upstream.json"),
        "rule":paper["clue_replication"],"histories":frozen,"model":model})
    if phase=="plan":
        return
    def run(item):
        sid,i,q,prompts=item
        arms=("native","source") if i%2 else ("source","native")
        result={a:measured_prompt(prompts[a],q,evaluator,model,cache,phase=="measure") for a in arms}
        return sid,str(i),result
    for name,selected in (("changed","native"),("unchanged","source")):
        file=fresh/(name+".json")
        if phase=="measure" and file.is_file():
            continue
        records={h:{} for h in histories}
        tasks=[r for r in work if frozen[r[0]][str(r[1])]["selected"]==selected]
        with ThreadPoolExecutor(max_workers=workers if phase=="measure" else 1) as pool:
            for sid,key,result in pool.map(run,tasks):
                records[sid][key]=result
        register(file,records)


def readback(base, out, corpus, split, phase, workers, fork=None, bank="native", clues=False, fresh_clues=False):
    from types import SimpleNamespace
    from experiments.controller import text_hash
    from experiments.controller.paper_f1 import PROTOCOL, load_split_cohort
    from experiments.controller.reader_slot_credit import archive_digests, measured_contexts, measured_prompt, registered_swap_contexts
    from src.eval.locomo import LoCoMoEvaluator

    paper = read(PROTOCOL)
    rule, dataset = paper["source_readback"], paper["datasets"]["locomo"]
    if clues and split == "test":
        decision = read(out/"clues/validation/upstream.json")
        if not decision["release_test"] or decision["rule"] != paper["clue_retention"] or decision["all"]["questions"] != 341:
            raise ValueError("Clue routing did not pass fixed validation")
    clue_dir = out/"clues"/("forks" if fork else split)
    if fork:
        clue_dir = clue_dir/fork/bank
    samples, questions = load_split_cohort("locomo", corpus, dataset, split)
    meta = read(base/"locomo/protocol.json")
    encode_ids, decode = readback_tokenizer(meta)
    if meta["encode_batch_size"] != 64:
        raise ValueError("Unexpected native LoCoMo encoder batch size")
    if ((split != "test" and meta.get("split") != split) or (split == "test" and meta.get("split","test") != "test")
            or len(questions) != (dataset["scored_questions"] if split=="test" else dataset["development_splits"][split]["scored_questions"])):
        raise ValueError("Native cohort mismatch")
    if split == "test":
        release = read(out/"validation/upstream.json")
        if not release["eligible"] or release["rule"] != rule or release["questions"] != 341 or release["evaluator"] != paper["upstream_revision"]:
            raise ValueError("Source readback requires a verified positive validation decision")
    fork_info = {}
    if fork:
        if split != "train" or fork not in paper["training_boundary_swap"]["histories"]:
            raise ValueError("A source-by-selection cross needs a registered training decision")
        cell = paper["training_boundary_swap"]["histories"][fork]
        samples = [sample for sample in samples if sample["sample_id"] == fork]
        questions = {k:q for k,q in questions.items() if k[0] == fork}
        if len(samples) != 1 or len(questions) != cell["questions"]:
            raise ValueError("Incomplete training history")
        marker = base.with_name(base.name+"-swap")/fork/f"span-{cell['span']}"/"locomo/swap"/fork/"measurement.json"
        if read(out/"train"/fork/"plan.json")["archive"] != archive_digests(base/"locomo/native"/fork):
            raise ValueError("The native fork archive changed since source registration")
        fork_info = {"fork":{"history":fork, "bank":bank, "measurement":file_hash(marker), "design":paper["fork_readback"]}}
        out = out/"forks"/fork/bank
    else:
        if bank != "native":
            raise ValueError("Swap bank requires a training fork")
        out = out/split
    out.mkdir(parents=True, exist_ok=True)
    ev = LoCoMoEvaluator(SimpleNamespace())
    source_cache = (base.with_name("llama31-paper-v1") if split == "test" else base)/"greedy-cache"
    fresh = (out.parent if fork else out)/"responses"
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(fresh)
    work, manifest = [], {}
    for sample in samples:
        sid = sample["sample_id"]
        native = base/"locomo/native"/sid
        qa = {i:q for (h,i),q in questions.items() if h == sid}
        model = meta["model"]["model"]
        if fork and bank == "swap":
            with next((native/"traces").glob("*.jsonl")).open() as stream:
                native_spans = [e for line in stream if (e:=json.loads(line))["kind"] == "span"]
            _, expected, data = registered_swap_contexts(marker.parent, native_spans, cell, qa, source_cache, ev, model)
        else:
            expected = archive_digests(native)
            data = measured_contexts(native, source_cache, expected, qa, ev, model)
        spans, contexts, archived = data
        turns = {t["dia_id"]: t["text"] for g in sample["conversation"].values() if isinstance(g,list) for t in g}
        coverage = {}
        plan = {}
        routes = {}
        route = clue_router(spans,paper["clue_retention"]) if clues else None
        for i, ctx, orig in zip(qa, contexts, archived, strict=True):
            question = qa[i]["question"]
            def build(texts):
                return ev.build_prompt(question, texts, {"question": question})
            prompts, (chosen,foil), excerpts = cited_contexts(ctx, spans, encode_ids, decode, build, rule)
            hashes = {a:text_hash(p) for a,p in prompts.items()}
            sizes = {a:len(encode_ids(p)) for a,p in prompts.items()}
            if hashes["native"] != orig["prompt_sha256"] or max(sizes.values()) != sizes["native"]:
                raise ValueError("Original response or readback token cap disagrees")
            plan[str(i)] = {"origins":[chosen,foil], "hashes":hashes, "prompt_tokens":sizes, "native_key":orig["key"]}
            work.append((sid,i,qa[i],prompts))
            if clues:
                routes[str(i)] = route(question,ctx,excerpts[0],rule["keep_native_ranks"])
            evidence = set(qa[i].get("evidence") or ())
            coverage[str(i)] = {"labelled": bool(evidence)}
            for arm, pos, text in zip(("source","sham"),(chosen,foil),excerpts,strict=True):
                known = evidence.intersection(spans[pos]["source_turns"])
                coverage[str(i)][arm] = {"origin": bool(known), "visible": any(turns[k] in text for k in known)}
        folder = out/sid
        folder.mkdir(exist_ok=True)
        register(folder/"plan.json", {"rule":rule,"archive":expected,"rows":plan})
        register(folder/"coverage.json", coverage)
        manifest[sid] = file_hash(folder/"plan.json")
        if clues:
            clue_dir.mkdir(parents=True,exist_ok=True)
            register(clue_dir/(sid+".json"),{"rule":paper["clue_retention"],"reference_plan":manifest[sid],"rows":routes})

    register(out/"plan.json", {"split":split,"model":meta["model"],"backend":meta["backend"],"questions":len(work),
                                "rule":rule,"histories":manifest,**fork_info,"native_protocol_sha256":file_hash(base/"locomo/protocol.json")})
    if fresh_clues:
        if split != "test" or fork or not clues:
            raise ValueError("Fresh clue evaluation is registered on the frozen LoCoMo test split")
        repeat_clue_reader(clue_dir,work,paper["clue_retention"],paper,meta["model"]["model"],ev,phase,workers)
        return
    if phase == "plan" or clues:
        return
    if split == "train" and not fork:
        raise ValueError("Training cohort is used only for coverage")
    generate = phase == "measure" and not (out/"summary.json").exists()
    os.environ["MEMSKILL_USAGE_LOG"] = str(out/"usage.jsonl")
    arms = ("native","source","sham")
    def run(args):
        ordinal, (sid,i,q,prompts) = args
        cut = ordinal % len(arms)
        answers = {arm:measured_prompt(prompts[arm],q,ev,meta["model"]["model"],fresh,generate)
                   for arm in arms[cut:]+arms[:cut]}
        if max(a["tokens"][0] for a in answers.values()) > answers["native"]["tokens"][0]:
            raise ValueError("Serving input budget exceeded")
        return sid, str(i), answers
    results = {sid:{} for sid in manifest}
    with ThreadPoolExecutor(max_workers=workers if generate else 1) as pool:
        for sid, key, answers in pool.map(run,enumerate(work)):
            results[sid][key] = answers
    groups = {}
    for sid, answers in results.items():
        path = out/sid/"answers.json"
        register(path, answers)
        groups[sid] = {"questions":len(answers),"sha256":file_hash(path),
                       "f1":{a:100*mean(r[a]["f1"] for r in answers.values()) for a in arms}}
    register(out/"summary.json", {"plan_sha256":file_hash(out/"plan.json"), "histories":groups})



def raw_turn_passages(history):
    processor = LoCoMoProcessor(chunk_mode="full-session")
    passages = {}
    sessions = sorted(int(key[8:]) for key in history if key.startswith("session_") and key[8:].isdigit())
    for index in sessions:
        key = f"session_{index}"
        turns = history[key]
        for i in range(len(turns)):
            group = turns[max(0,i-1):i+2]
            text, = processor._extract_session_chunks(group, history[key+"_date_time"])
            passages.setdefault(text, tuple(t["dia_id"] for t in group))
    return list(passages.items())


def dense_raw_ranks(out, corpus):
    """Rank question-to-raw passages on the leased GPU. No answer references enter the encoder."""
    from experiments.controller.paper_f1 import PROTOCOL, load_split_cohort

    paper = read(PROTOCOL)
    design = paper["direct_raw_retrieval"]
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or not os.environ.get("GPU_GUARD_LEASE_FDS")
            or not 0 < float(os.environ["MEMSKILL_ENCODER_DUTY"]) <= .8):
        raise RuntimeError("Raw passage scoring requires one leased CUDA encoder")
    model_meta = out.parent.parent/"llama31-paper-v1/verified-qwen.json"
    checkpoint = Path(os.environ["MEMSKILL_QWEN_EMBEDDING_PATH"])
    manifest = read(model_meta)
    if manifest["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3" or any(
        file_hash(checkpoint / record["file"]) != record["sha256"] for record in manifest["verified_files"]
    ):
        raise ValueError("Direct retrieval requires the verified original Qwen checkpoint")
    set_retriever_flash_attn(False)
    _, encoder = init_context_model("qwen3-embedding-0.6b")
    if not all(p.is_cuda for p in encoder.parameters()):
        raise RuntimeError("Qwen is not on a GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    def vectors(texts):
        return F.normalize(torch.as_tensor(encode_batch(encoder,texts,8,normalize=True),device="cuda").float(),dim=1)
    for split in design["cohorts"]:
        histories, qa = load_split_cohort("locomo",corpus,paper["datasets"]["locomo"],split)
        for h in histories:
            sid = h["sample_id"]
            target = out/"direct"/split/sid
            target.mkdir(parents=True,exist_ok=True)
            p = target/"ranks.json"
            passages = raw_turn_passages(h["conversation"])
            indices = [i for key,i in qa if key == sid]
            questions = [qa[sid,i]["question"] for i in indices]
            inputs = {"passage_sha256":digest(passages),"question_sha256":digest(list(zip(indices,questions))),
                      "corpus_sha256":file_hash(corpus),"qwen_sha256":file_hash(model_meta),"design":design}
            if p.exists():
                if read(p)["inputs"] != inputs or read(p)["cosines_sha256"] != file_hash(target/"cosines.npy"):
                    raise ValueError("Earlier direct ranks used different inputs")
                continue
            docs = vectors([t for t,_ in passages])
            queries = vectors(questions)
            scores = queries @ docs.T
            if docs.shape[1] != 1024 or not torch.isfinite(scores).all():
                raise ValueError("Invalid raw passage scores")
            with TemporaryDirectory(dir=target) as temp:
                matrix = Path(temp)/"cosines.npy"
                np.save(matrix,scores.cpu().numpy(),allow_pickle=False)
                os.replace(matrix,target/"cosines.npy")
            # Persist ordered top twenty for replay. All similarities were computed on CUDA.
            order = torch.argsort(scores,descending=True,stable=True)[:,:20]
            top = torch.gather(scores,1,order).tolist()
            ranks = {str(i):{"ids":ids,"scores":values} for i,ids,values in zip(indices,order.tolist(),top,strict=True)}
            register(p,{"inputs":inputs,"cosines_sha256":file_hash(target/"cosines.npy"),"passages":len(passages),"raw_utf8_bytes":sum(len(t.encode()) for t,_ in passages),
                "questions":len(questions),"dimension":docs.shape[1],"device":torch.cuda.get_device_name(),"ranks":ranks})
            print(split,sid,"ranked",len(questions),"against",len(passages),"raw passages",flush=True)


def fit_raw_passages(start, ranked, build, encode_ids, decode, limit, first_only=False):
    lines = list(start)
    seen = set()
    used = []
    for key, (text, ids) in ranked:
        if seen.intersection(ids):
            continue
        tokens = encode_ids(text)
        n = min(len(tokens),256) if first_only else len(tokens)
        while n > 0:
            excerpt = text if n == len(tokens) else decode(tokens[:n]).strip()
            excess = len(encode_ids(build([*lines,excerpt]))) - limit
            if excess <= 0:
                lines.append(excerpt)
                seen.update(ids)
                used.append(key)
                break
            n -= max(1,excess)
        if first_only or n < len(tokens):
            break
    prompt = build(lines)
    if len(encode_ids(prompt)) > limit:
        raise ValueError("Direct reading exceeded the reference input budget")
    return prompt, used


def direct_reader(base, out, corpus, split, sid, phase, workers):
    from types import SimpleNamespace
    from experiments.controller import text_hash
    from experiments.controller.paper_f1 import PROTOCOL, load_split_cohort
    from experiments.controller.reader_slot_credit import archive_digests, measured_contexts, measured_prompt
    from src.eval.locomo import LoCoMoEvaluator

    paper = read(PROTOCOL)
    design = paper["direct_raw_retrieval"]
    if split not in design["cohorts"]:
        raise ValueError("Unknown direct-reading cohort")
    samples,cohort = load_split_cohort("locomo",corpus,paper["datasets"]["locomo"],split)
    sample, = (h for h in samples if h["sample_id"] == sid)
    qa = {i:q for (h,i),q in cohort.items() if h == sid}
    ref = out/split
    certified = read(ref/"upstream.json")
    if certified["evaluator"] != paper["upstream_revision"] or certified["rule"] != paper["source_readback"]:
        raise ValueError("Pointer panel has no matching author-score receipt")
    source_plan = read(ref/sid/"plan.json")
    original = read(ref/sid/"answers.json")
    trial = out/"direct"/split/sid
    rankfile = trial/"ranks.json"
    ranking = read(rankfile)
    passages = raw_turn_passages(sample["conversation"])
    meta = read(base/"locomo/protocol.json")
    encode_ids,decode = readback_tokenizer(meta)
    native = base/"locomo/native"/sid
    archive = archive_digests(native)
    rank_inputs = ranking["inputs"]
    if (rank_inputs["design"] != design or rank_inputs["corpus_sha256"] != file_hash(corpus)
            or rank_inputs["passage_sha256"] != digest(passages)
            or rank_inputs["question_sha256"] != digest([(i,q["question"]) for i,q in qa.items()])
            or ranking["passages"] != len(passages) or ranking["questions"] != len(qa)
            or rank_inputs["qwen_sha256"] != file_hash(out.parent.parent/"llama31-paper-v1/verified-qwen.json")
            or ranking["dimension"] != 1024 or "3090" not in ranking["device"]
            or archive != source_plan["archive"] or meta["encode_batch_size"] != 64):
        raise ValueError("Raw or native input receipt changed")
    model = meta["model"]["model"]
    ev = LoCoMoEvaluator(SimpleNamespace())
    native_cache = (base if split=="validation" else base.with_name("llama31-paper-v1"))/"greedy-cache"
    cache = trial/"responses"
    os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
    _,contexts,archived = measured_contexts(native,native_cache,archive,qa,ev,model)
    cosines = np.load(trial/"cosines.npy",allow_pickle=False)
    if (ranking["cosines_sha256"] != file_hash(trial/"cosines.npy") or cosines.shape != (len(qa),len(passages))
            or cosines.dtype != np.float32 or not np.isfinite(cosines).all()):
        raise ValueError("GPU cosine receipt is missing or changed")
    top_ids = np.argsort(-cosines,axis=1,kind="stable")[:,:20]
    for row,i in enumerate(qa):
        top = ranking["ranks"][str(i)]
        if top["ids"]!=top_ids[row].tolist() or top["scores"]!=cosines[row,top_ids[row]].tolist():
            raise ValueError("Published raw ranks do not match the GPU scores")
    rows,work = {},[]
    os.environ["MEMSKILL_USAGE_LOG"] = str(trial/"usage.jsonl")
    for i,context,old in zip(qa,contexts,archived,strict=True):
        key = str(i)
        answers = original[key]
        ref_row = source_plan["rows"][key]
        if answers["native"]["prompt_sha256"] != old["prompt_sha256"]:
            raise ValueError("Original reader prompt changed")
        rank = ranking["ranks"][key]
        ids, scores = rank["ids"],rank["scores"]
        if (len(ids)!=len(set(ids)) or len(ids)!=len(scores) or len(ids)!=20
                or any(v<0 or v>=len(passages) for v in ids) or sorted(scores,reverse=True)!=scores):
            raise ValueError("Invalid raw passage rank")
        ranked = [(n,passages[n]) for n in ids]
        q = qa[i]
        build = lambda texts: ev.build_prompt(q["question"],texts,q)
        kept = [t for _,t in context[:paper["source_readback"]["keep_native_ranks"]]]
        limit = ref_row["prompt_tokens"]["source"]
        direct, u = fit_raw_passages(kept,ranked,build,encode_ids,decode,limit,True)
        raw, v = fit_raw_passages([],ranked,build,encode_ids,decode,limit)
        prompts = {"direct":direct,"raw":raw}
        rows[key] = {"hashes":{a:text_hash(p) for a,p in prompts.items()},
            "sizes":{a:len(encode_ids(p)) for a,p in prompts.items()},"limit":limit,
            "used":{"direct":u,"raw":v}}
        work.append((key,q,prompts,answers))
    register(trial/"plan.json",{"design":design,"split":split,"history":sid,"rows":rows,"rank_sha256":file_hash(rankfile),
        "reference_sha256":file_hash(ref/"upstream.json"),"reference_answers_sha256":file_hash(ref/sid/"answers.json")})
    if phase == "plan":
        return
    generate = phase=="measure" and not (trial/"answers.json").is_file()
    def run(item):
        key,q,prompts,prior = item
        result = {a:prior[a] for a in ("native","source")}
        by_hash = {r["prompt_sha256"]:r for r in prior.values()}
        for a in (("direct","raw") if int(key)%2 else ("raw","direct")):
            p = prompts[a]
            result[a] = by_hash.get(text_hash(p)) or measured_prompt(p,q,ev,model,cache,generate)
            if result[a]["tokens"][0] > prior["source"]["tokens"][0]:
                raise ValueError("Measured direct input budget changed")
        return key,result
    with ThreadPoolExecutor(max_workers=workers if generate else 1) as pool:
        register(trial/"answers.json",dict(pool.map(run,work)))


def probe_marker_reading(base, out, corpus, sid, phase, workers):
    from types import SimpleNamespace
    from experiments.controller import text_hash
    from experiments.controller.paper_f1 import PROTOCOL, load_split_cohort
    from experiments.controller.reader_slot_credit import archive_digests, corrected_marker_contents, measured_contexts, measured_prompt
    from src.eval.locomo import LoCoMoEvaluator

    paper=read(PROTOCOL)
    _, cohort=load_split_cohort("locomo",corpus,paper["datasets"]["locomo"],"train")
    qa={i:q for (h,i),q in cohort.items() if h==sid}
    root=out/"forks"/sid
    crossing=read(root/"upstream.json")
    meta=read(base/"locomo/protocol.json")
    if (crossing["questions"]!=len(qa) or crossing["scorer_commit"]!=paper["upstream_revision"]
            or crossing["design"]!=paper["fork_readback"] or meta["encode_batch_size"]!=64):
        raise ValueError("Expected an independently scored training cross")
    encode_ids,_=readback_tokenizer(meta);model=meta["model"]["model"]
    history=base/"locomo/native"/sid
    trial=out/"writer-format"/sid; trial.mkdir(parents=True,exist_ok=True)
    cache=trial/"responses"; os.environ["MEMSKILL_GREEDY_EVAL_CACHE"]=str(cache)
    os.environ["MEMSKILL_USAGE_LOG"]=str(trial/"usage.jsonl")
    ev=LoCoMoEvaluator(SimpleNamespace()); archive=archive_digests(history)
    spans,ctx,old=measured_contexts(history,base/"greedy-cache",archive,qa,ev,model)
    lookup,counts=corrected_marker_contents(spans)
    trace,=(history/"traces").glob("*.jsonl")
    refs={e["query_id"]:e["refs"] for line in trace.open() if (e:=json.loads(line))["kind"]=="qa"}
    endpoints=read(root/"native"/sid/"answers.json"); rows={}; work=[]
    for i,context,orig in zip(qa,ctx,old,strict=True):
        q=qa[i]; plain=[t for _,t in context]
        variants=[lookup.get(tuple(ref),(t,t)) for ref,t in zip(refs[i],plain,strict=True)]
        texts={"native":plain,"clean":[v[0] for v in variants],"neutral":[v[1] for v in variants]}
        prompts={a:ev.build_prompt(q["question"],ts,q) for a,ts in texts.items()}
        hashes={a:text_hash(p) for a,p in prompts.items()};sizes={a:len(encode_ids(p)) for a,p in prompts.items()}
        if hashes["native"]!=orig["prompt_sha256"] or sizes["clean"]>sizes["native"] or sizes["neutral"]!=sizes["native"]:
            raise ValueError("Format probe changed original prompt or neutral token budget")
        rows[str(i)]={"hashes":hashes,"changed_ranks":sum(a!=b for a,b in zip(plain,texts["clean"],strict=True)),
                      "prompt_tokens":sizes}
        work.append((i,prompts,q))
    register(trial/"plan.json",{"design":paper["writer_marker_probe"],"history":sid,"archive":archive,"parser":counts,
             "cross_sha256":file_hash(root/"upstream.json"),"rows":rows})
    if phase=="plan":return
    generate=phase=="measure" and not (trial/"answers.json").exists()
    def run(item):
        i,prompts,q=item
        normal=measured_prompt(prompts["native"],q,ev,model,root/"responses",False)
        if normal!=endpoints[str(i)]["native"]:
            raise ValueError("Plain reader endpoint changed since the selection cross")
        results={"native":normal}
        for arm in (("clean","neutral") if i%2 else ("neutral","clean")):
            results[arm]=normal if prompts[arm]==prompts["native"] else measured_prompt(prompts[arm],q,ev,model,cache,generate)
        if results["neutral"]["tokens"][0]!=normal["tokens"][0] or results["clean"]["tokens"][0]>normal["tokens"][0]:
            raise ValueError("Physical format token budgets differ")
        return str(i),results
    with ThreadPoolExecutor(max_workers=workers if generate else 1) as pool:
        register(trial/"answers.json",dict(pool.map(run,work)))


def transfer_readback(base, out, corpus, config, phase, clues=False):
    """Apply the frozen reader to each physically complete LongMemEval native bank."""
    from types import SimpleNamespace
    from experiments.controller import text_hash
    from experiments.controller.paper_f1 import PROTOCOL, load_split_cohort
    from experiments.controller.reader_slot_credit import archive_digests, measured_contexts, measured_prompt
    from src.eval.longmemeval import LongMemEvalEvaluator

    paper = read(PROTOCOL)
    spec, design = paper["datasets"]["longmemeval"], paper["readback_transfer"]
    setting = design["cohorts"][config]
    samples, cohort = load_split_cohort("longmemeval", corpus, spec, "test")
    meta = read(base/"longmemeval/protocol.json")
    if (meta["dataset"] != "longmemeval" or meta["test_keys"] != spec["scored_key_sha256"]
            or meta["encode_batch_size"] != setting["encode_batch_size"]
            or meta["settings"]["action_top_k"] != setting["action_top_k"]
            or meta["settings"]["mem_top_k_eval"] != 20 or len(cohort) != 101):
        raise ValueError("Wrong transfer settings or question population")
    encode_ids, decode = readback_tokenizer(meta)
    ev = LongMemEvalEvaluator(SimpleNamespace())
    limit = ev.prepare_eval_args().max_new_tokens
    if limit != 512:
        raise ValueError("Unexpected native LongMemEval output limit")
    model = meta["model"]["model"]
    study = out/"longmemeval"/config
    study.mkdir(parents=True,exist_ok=True)
    register(study/"plan.json", {"cohort_sha256":meta["test_keys"], "histories":list(s for s,i in cohort),
             "config":config,"design":design,"rule":paper["source_readback"],"protocol_sha256":file_hash(base/"longmemeval/protocol.json"),
             "model":meta["model"],"backend":meta["backend"],"max_tokens":limit})
    rank = int(os.environ.get("MEMSKILL_TRANSFER_RANK","0"))
    shards = int(os.environ.get("MEMSKILL_TRANSFER_SHARDS","1"))
    if not 0 <= rank < shards:
        raise ValueError("Invalid LongMemEval reader shard")
    for ordinal, sample in enumerate(s for s in samples if (s["question_id"],0) in cohort):
        if ordinal % shards != rank:
            continue
        sid = sample["question_id"]
        history, dest = base/"longmemeval/native"/sid, study/sid
        if not (history/"audit.json").is_file() or (phase=="verify" and not (dest/"answers.json").is_file()):
            continue
        if phase=="measure" and (dest/"answers.json").is_file():
            continue
        dest.mkdir(exist_ok=True)
        cache = dest/"responses"
        os.environ["MEMSKILL_GREEDY_EVAL_CACHE"] = str(cache)
        os.environ["MEMSKILL_USAGE_LOG"] = str(dest/"usage.jsonl")
        expected = archive_digests(history)
        spans, contexts, archived = measured_contexts(history,base/"greedy-cache",expected,{0:sample},ev,model,limit)
        def build(texts):
            return ev.build_prompt(sample["question"], texts, {"question_date":sample.get("question_date","")})
        prompts, origins, excerpts = cited_contexts(contexts[0],spans,encode_ids,decode,build,paper["source_readback"])
        hashes = {a:text_hash(p) for a,p in prompts.items()}
        sizes = {a:len(encode_ids(p)) for a,p in prompts.items()}
        if hashes["native"] != archived[0]["prompt_sha256"] or max(sizes.values()) != sizes["native"]:
            raise ValueError("LongMemEval native input or token budget changed")
        register(dest/"plan.json", {"study_sha256":file_hash(study/"plan.json"),"question":sid,"archive":expected,
                 "native_key":archived[0]["key"],"hashes":hashes,"prompt_tokens":sizes,"origins":list(origins)})
        if clues:
            route = clue_router(spans,paper["clue_retention"])(sample["question"],contexts[0],excerpts[0],paper["source_readback"]["keep_native_ranks"])
            register(dest/"clue-route.json",{"rule":paper["clue_retention"],"transfer":paper["clue_longmemeval"],
                "plan":file_hash(dest/"plan.json"),"route":route})
            continue
        if phase=="plan":
            continue
        arms = ("native","source","sham")
        cut = ordinal % len(arms)
        answers={a:measured_prompt(prompts[a],sample,ev,model,cache,phase=="measure",limit) for a in arms[cut:]+arms[:cut]}
        if max(v["tokens"][0] for v in answers.values()) > answers["native"]["tokens"][0]:
            raise ValueError("Transfer serving input budget exceeded")
        register(dest/"answers.json", answers)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("base", "study", "corpus"):
        p.add_argument(k, type=Path)
    p.add_argument("--readback", choices=["train","validation","test"])
    p.add_argument("--fresh-clues",action="store_true")
    p.add_argument("--clues",action="store_true")
    p.add_argument("--rank-direct", action="store_true")
    p.add_argument("--direct", nargs=2, metavar=("SPLIT","HISTORY"))
    p.add_argument("--marker-history")
    p.add_argument("--transfer-readback", choices=["primary","paper"])
    p.add_argument("--fork", help="Registered training history for the selection-by-reader cross")
    p.add_argument("--bank", choices=["native","swap"], default="native")
    p.add_argument("--phase", choices=["plan","measure","verify"], default="plan")
    p.add_argument("--workers", type=int, default=4)
    o = p.parse_args()
    if o.rank_direct:
        dense_raw_ranks(o.study,o.corpus)
    elif o.direct:
        direct_reader(o.base,o.study,o.corpus,*o.direct,o.phase,o.workers)
    elif o.marker_history:
        probe_marker_reading(o.base,o.study,o.corpus,o.marker_history,o.phase,o.workers)
    elif o.transfer_readback:
        transfer_readback(o.base, o.study, o.corpus, o.transfer_readback, o.phase,o.clues)
    elif o.readback:
        readback(o.base, o.study, o.corpus, o.readback, o.phase, o.workers, o.fork, o.bank, o.clues or o.fresh_clues,o.fresh_clues)
    else:
        encode(o.base, o.study, o.corpus)
