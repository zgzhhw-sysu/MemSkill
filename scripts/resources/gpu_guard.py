#!/usr/bin/env python3
"""Cooperative GPU memory reservation. Uses the CUDA driver, no PyTorch needed."""
import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve()
DEFAULT_STATE = Path.home() / ".local/state/memskill/gpu-guard"
MIB = 1024 * 1024


def atomic_json(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return default


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    return bool(record and identity(record["pid"]) == record)


def query_gpus(ids=None):
    out = subprocess.check_output([
        "nvidia-smi", *(["-i", ",".join(map(str, ids))] if ids else []), "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    devices = {}
    for line in out.splitlines():
        idx, uuid, total, used, free, util = [s.strip() for s in line.split(",")]
        devices[int(idx)] = {"uuid": uuid, "total_mib": int(total),
                             "used_mib": int(used), "free_mib": int(free), "utilization": int(util)}
    return devices


def query_processes(ids=None):
    out = subprocess.check_output([
        "nvidia-smi", *(["-i", ",".join(map(str, ids))] if ids else []), "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    processes = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        uuid, pid = [s.strip() for s in line.split(",")]
        processes.setdefault(uuid, set()).add(int(pid) if pid.isdigit() else -1)
    return processes


def gpu_list(value):
    ids = sorted({int(v) for v in value.split(",")})
    if not ids or min(ids) < 0:
        raise argparse.ArgumentTypeError("GPU IDs must be nonnegative integers")
    return ids


def lease_path(state, gpu):
    return state / f"gpu-{gpu}.lease"


def lease_busy(state, gpu):
    with lease_path(state, gpu).open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def ready_path(state, gpu):
    return state / f"gpu-{gpu}.ready.json"


def unit_name(state):
    digest = hashlib.sha256(str(state).encode()).hexdigest()[:10]
    return f"memskill-gpu-guard-{digest}.service"


def unit_path(state):
    return Path.home() / ".config/systemd/user" / unit_name(state)


def daemon_status(state):
    status = read_json(state / "status.json", {})
    if not alive(status.get("daemon")):
        return None
    return status


def systemd_quote(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def start(state, args):
    with (state / "control.lock").open("a+") as control:
        fcntl.flock(control, fcntl.LOCK_EX)
        if daemon_status(state):
            print("GPU guard is already running; use status / release / hold / stop.", flush=True)
            return
        old = read_json(state / "config.json", {})
        ids = args.gpus or old.get("gpus", [4, 5, 6, 7])
        inventory = query_gpus(ids)
        missing = set(ids) - inventory.keys()
        if missing:
            raise RuntimeError(f"Unknown GPUs: {sorted(missing)}")
        reserve = args.reserve_mib if args.reserve_mib is not None else old.get("reserve_mib", 4096)
        hold = args.hold_mib if args.hold_mib is not None else old.get("hold_mib", 2048) or 2048
        if reserve < 128 or hold < 0:
            raise RuntimeError("reserve-mib must be >= 128; hold-mib must be >= 0")
        shared = getattr(args, "shared", None)
        config = {"gpus": ids, "devices": {str(i): inventory[i]["uuid"] for i in ids},
                  "reserve_mib": reserve, "hold_mib": hold, "poll_seconds": 1.0,
                  "shared": shared if shared is not None else old.get("shared", False)}
        atomic_json(state / "config.json", config)
        unit = unit_path(state)
        unit.parent.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-u", SCRIPT, "--state-dir", state, "_serve"]
        unit.write_text("[Unit]\nDescription=MemSkill cooperative GPU memory guard\n\n"
                        "[Service]\nType=simple\nExecStart=" + " ".join(map(systemd_quote, argv)) +
                        "\nRestart=on-failure\nRestartSec=5\nKillMode=control-group\n"
                        "TimeoutStopSec=20\nUMask=0077\n\n[Install]\nWantedBy=default.target\n")
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", unit.name], check=True)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if daemon_status(state):
                print(f"Started {unit.name}; GPUs {ids}.", flush=True)
                return
            time.sleep(0.2)
        raise RuntimeError(f"Startup timed out; inspect journalctl --user -u {unit.name}")


def stop(state):
    if unit_path(state).exists():
        subprocess.run(["systemctl", "--user", "disable", "--now", unit_name(state)], check=True)
    else:
        status = daemon_status(state)
        if status:
            os.kill(status["daemon"]["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 20
            while alive(status["daemon"]) and time.monotonic() < deadline:
                time.sleep(0.1)
            if alive(status["daemon"]):
                raise RuntimeError("Daemon did not stop within 20 seconds")
    print("GPU guard stopped. Experiment processes are not stopped.")


def selected_gpus(state, requested):
    cfg = read_json(state / "config.json")
    if not cfg:
        raise RuntimeError("Run start first")
    ids = requested or cfg["gpus"]
    if not set(ids) <= set(cfg["gpus"]):
        raise RuntimeError(f"Requested GPUs must belong to configured GPUs {cfg['gpus']}")
    return cfg, ids


def pause(state, requested, release):
    _, ids = selected_gpus(state, requested)
    with (state / "control.lock").open("a+") as control:
        fcntl.flock(control, fcntl.LOCK_EX)
        paused = set(read_json(state / "paused.json", []))
        paused = paused.union(ids) if release else paused.difference(ids)
        atomic_json(state / "paused.json", sorted(paused))
    if release and daemon_status(state):
        wait_released(state, ids, require_empty=False)
    print(f"{'Released' if release else 'Resumed holding'} GPUs {ids}.")


def wait_released(state, ids, require_empty=True):
    deadline = time.monotonic() + 35
    requested_at = time.time()
    foreign = {}
    while time.monotonic() < deadline:
        status = daemon_status(state)
        if not status:
            raise RuntimeError("GPU guard is not running")
        if status["updated_at"] >= requested_at and all(
                status.get("gpus", {}).get(str(i), {}).get("state") in ("leased", "paused")
                for i in ids):
            if not require_empty:
                return
            # Wait for actual CUDA contexts to disappear, not only process termination.
            cfg = read_json(state / "config.json")
            apps = query_processes(ids)
            foreign = {i: sorted(apps.get(cfg["devices"][str(i)], set())) for i in ids
                       if apps.get(cfg["devices"][str(i)])}
            if not foreign:
                return
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for GPUs to clear; processes={foreign}. No other jobs were killed.")


def run_job(state, args):
    cfg, ids = selected_gpus(state, args.gpus)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise RuntimeError("Supply a command after --")
    inherited = os.environ.get("GPU_GUARD_LEASE_FDS")
    if inherited:
        raise RuntimeError("Nested gpu_guard run is unsupported; launch the child command directly")
    if not daemon_status(state):
        start(state, argparse.Namespace(gpus=None, reserve_mib=None, hold_mib=None))
    with contextlib.ExitStack() as stack:
        leases = []
        print(f"Waiting for GPU leases {ids}...", flush=True)
        # Acquire in ascending order so overlapping multi-GPU jobs cannot deadlock.
        for gpu in ids:
            f = stack.enter_context(lease_path(state, gpu).open("a+"))
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            f.truncate()
            f.write(json.dumps({"owner": identity(os.getpid()), "gpus": ids}))
            f.flush()
            leases.append(f)
        if cfg.get("shared"):
            wait_released(state, cfg["gpus"], require_empty=False)
        else:
            wait_released(state, ids)
        env = dict(os.environ)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(cfg["devices"][str(i)] for i in ids)
        if args.cuda_ordinals:
            # CUDA_DEVICE_ORDER=PCI_BUS_ID sorts visible CUDA devices by PCI address.
            rows = subprocess.check_output(["nvidia-smi", "--query-gpu=pci.bus_id,uuid",
                                            "--format=csv,noheader"], text=True).splitlines()
            ordered = [line.split(",")[1].strip() for line in sorted(rows)]
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(ordered.index(cfg["devices"][str(i)])) for i in ids)
        env["GPU_GUARD_LEASE_FDS"] = ",".join(str(f.fileno()) for f in leases)
        child = subprocess.Popen(command, env=env, start_new_session=True,
                                 pass_fds=tuple(f.fileno() for f in leases))
        previous = {}

        def forward(signum, _frame):
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, forward)
        try:
            print(f"GPUs released; job PID {child.pid}; physical {ids} -> CUDA 0..{len(ids)-1}.", flush=True)
            rc = child.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            # Closing our descriptors releases leases only after inherited copies close too.
        print(f"Job exited ({rc}); automatic holding resumes when leases and CUDA processes clear.", flush=True)
        return rc if rc >= 0 else 128 - rc


def cuda_worker(state, gpu):
    cfg = read_json(state / "config.json")
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["devices"][str(gpu)]
    parent = os.getppid()
    libc = ctypes.CDLL(None)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise RuntimeError("Unable to arrange cleanup on daemon exit")
    running = True

    def shutdown(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    if parent != os.getppid():
        return
    cuda = ctypes.CDLL("libcuda.so.1")
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuCtxCreate_v2": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int],
        "cuCtxDestroy_v2": [ctypes.c_void_p],
        "cuMemGetInfo_v2": [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)],
        "cuMemAlloc_v2": [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
        "cuMemFree_v2": [ctypes.c_uint64],
    }
    for name, signature in signatures.items():
        getattr(cuda, name).argtypes = signature
        getattr(cuda, name).restype = ctypes.c_int

    def call(name, *args):
        rc = getattr(cuda, name)(*args)
        if rc:
            raise RuntimeError(f"{name}: CUDA error {rc}")

    ctx = ctypes.c_void_p()
    ptr = ctypes.c_uint64()
    try:
        call("cuInit", 0)
        dev = ctypes.c_int()
        call("cuDeviceGet", ctypes.byref(dev), 0)
        call("cuCtxCreate_v2", ctypes.byref(ctx), 4, dev)  # CU_CTX_SCHED_BLOCKING_SYNC
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        call("cuMemGetInfo_v2", ctypes.byref(free), ctypes.byref(total))
        size = free.value - cfg["reserve_mib"] * MIB
        size = min(size, (cfg["hold_mib"] or 2048) * MIB)
        size = size // MIB * MIB
        if size < MIB:
            raise RuntimeError("Insufficient free memory; will wait instead of displacing another job")
        if not running:
            return
        call("cuMemAlloc_v2", ctypes.byref(ptr), size)
        atomic_json(ready_path(state, gpu), {"worker": identity(os.getpid()),
                                          "allocated_mib": size // MIB})
        print(f"GPU {gpu}: holding {size // MIB} MiB, no compute loop", flush=True)
        while running:
            time.sleep(0.5)
    finally:
        if ptr.value:
            cuda.cuMemFree_v2(ptr)
        if ctx.value:
            cuda.cuCtxDestroy_v2(ctx)


def serve(state):
    lock = (state / "daemon.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another daemon is already running")
    cfg = read_json(state / "config.json")
    if not cfg:
        raise RuntimeError("Missing config.json")
    inventory = query_gpus(cfg["gpus"])
    for gpu in cfg["gpus"]:
        if inventory.get(gpu, {}).get("uuid") != cfg["devices"][str(gpu)]:
            raise RuntimeError("GPU indices changed; stop and start to refresh the configured mapping")
    workers, retry_at = {}, {}
    running = True

    def shutdown(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    def stop_worker(gpu):
        p = workers.pop(gpu, None)
        if p:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
            p.guard_log.close()
        ready_path(state, gpu).unlink(missing_ok=True)

    try:
        while running:
            status = {"daemon": identity(os.getpid()), "updated_at": time.time(), "gpus": {}}
            try:
                apps = query_processes(cfg["gpus"])
                if cfg.get("shared"):
                    inventory = query_gpus(cfg["gpus"])
                query_error = None
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                apps, query_error = {}, str(exc)
            paused = set(read_json(state / "paused.json", []))
            leased = {gpu for gpu in cfg["gpus"] if lease_busy(state, gpu)}
            for gpu in cfg["gpus"]:
                entry = {"uuid": cfg["devices"][str(gpu)]}
                p = workers.get(gpu)
                if p and p.poll() is not None:
                    entry["last_exit"] = p.returncode
                    stop_worker(gpu)
                    retry_at[gpu] = time.monotonic() + 10
                    p = None
                foreign = apps.get(entry["uuid"], set()) - ({p.pid} if p else set())
                if gpu in leased:
                    stop_worker(gpu)
                    entry["state"] = "leased"
                elif gpu in paused or leased:
                    stop_worker(gpu)
                    entry["state"] = "paused"
                elif query_error:
                    stop_worker(gpu)
                    entry.update(state="query_error", error=query_error)
                elif cfg.get("shared") and inventory[gpu]["free_mib"] < (
                        cfg["reserve_mib"] + (0 if p else cfg["hold_mib"] + 512)):
                    stop_worker(gpu)
                    entry.update(state="waiting_for_memory", free_mib=inventory[gpu]["free_mib"])
                elif foreign and not cfg.get("shared"):
                    stop_worker(gpu)
                    entry.update(state="waiting_for_processes", pids=sorted(foreign))
                elif time.monotonic() < retry_at.get(gpu, 0):
                    entry["state"] = "retry_wait"
                else:
                    if p is None and running:
                        ready_path(state, gpu).unlink(missing_ok=True)
                        log = (state / f"gpu-{gpu}.log").open("ab", buffering=0)
                        p = subprocess.Popen([sys.executable, "-u", str(SCRIPT), "--state-dir",
                                              str(state), "_worker", "--gpu", str(gpu)],
                                             stdout=log, stderr=subprocess.STDOUT)
                        p.guard_log = log
                        workers[gpu] = p
                    ready = read_json(ready_path(state, gpu))
                    if p and ready and ready.get("worker") == identity(p.pid):
                        entry.update(state="holding", **ready)
                    else:
                        entry["state"] = "allocating"
                status["gpus"][str(gpu)] = entry
            atomic_json(state / "status.json", status)
            time.sleep(cfg.get("poll_seconds", 1))
    finally:
        for gpu in list(workers):
            stop_worker(gpu)
        atomic_json(state / "status.json", {"daemon": None, "gpus": {}, "updated_at": time.time()})
        lock.close()


def show_status(state, as_json):
    status = daemon_status(state)
    if as_json:
        print(json.dumps(status or {"daemon": None}, ensure_ascii=False, indent=2))
        return
    if not status:
        print("GPU guard is stopped.")
        return
    devices = query_gpus(read_json(state / "config.json")["gpus"])
    age = time.time() - status["updated_at"]
    print(f"Daemon PID {status['daemon']['pid']}; heartbeat age {age:.1f}s; {unit_name(state)}")
    for idx, item in status["gpus"].items():
        d = devices.get(int(idx), {})
        print(f"GPU {idx}: {item['state']:22s} allocated={item.get('allocated_mib', 0):5d} MiB "
              f"device_used={d.get('used_mib', '?')} MiB util={d.get('utilization', '?')}% "
              f"holder_pid={item.get('worker', {}).get('pid', '-')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("start", help="Install and start a persistent user service")
    p.add_argument("--gpus", type=gpu_list)
    p.add_argument("--shared", action=argparse.BooleanOptionalAction, default=None,
                   help="Allow other CUDA processes; only hold idle memory above the reserve floor")
    p.add_argument("--reserve-mib", type=int)
    p.add_argument("--hold-mib", type=int, help="Idle allocation cap per GPU in MiB (default 2048; 0 uses default)")
    p = commands.add_parser("status")
    p.add_argument("--json", action="store_true")
    commands.add_parser("stop", help="Stop holding; does not terminate experiments")
    for name in ("release", "hold"):
        commands.add_parser(name).add_argument("--gpus", type=gpu_list)
    p = commands.add_parser("run", help="Release GPUs, run command, automatically resume holding")
    p.add_argument("--gpus", type=gpu_list)
    p.add_argument("--cuda-ordinals", action="store_true", help="Use PCI-ordered numeric visibility for older vLLM versions")
    p.add_argument("command", nargs=argparse.REMAINDER)
    commands.add_parser("_serve", help=argparse.SUPPRESS)
    commands.add_parser("_worker", help=argparse.SUPPRESS).add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    state = args.state_dir.expanduser().resolve()
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(state, 0o700)
    if args.action == "start":
        start(state, args)
    elif args.action == "stop":
        stop(state)
    elif args.action == "status":
        show_status(state, args.json)
    elif args.action in ("release", "hold"):
        pause(state, args.gpus, args.action == "release")
    elif args.action == "run":
        return run_job(state, args)
    elif args.action == "_serve":
        serve(state)
    elif args.action == "_worker":
        cuda_worker(state, args.gpu)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"gpu_guard: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
