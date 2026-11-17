"""Generate or resume a reproducible pilot or full magnno value dataset."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from heapq import heapify, heappop, heappush
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from selfplay import DEFAULT_MAX_TURNS
from synthetic_game import SamplingConfig, simulate
from synthetic_plan import PlanConfig, make_plan, spec_for
from synthetic_store import (SCHEMA_VERSION, SPLITS, append_game, digest, export, increment,
                             journal, json_text, write_json)

HERE = Path(__file__).resolve().parent
CORE = ("game_state.py","backgammon.py","features.py","heuristic.py","agents/policies.py","agents/__init__.py")
GENERATOR = ("selfplay.py","simulation.py","synthetic_game.py","synthetic_plan.py","training.py","synthetic_data.py","synthetic_store.py")

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def file_set_digest(names):
    value = sha256()
    for name in names:
        value.update(name.encode() + b"\0" + bytes.fromhex(digest(HERE/name)))
    return value.hexdigest()

def software():
    import numpy
    import pyarrow
    cmd = subprocess.run(["git","rev-parse","HEAD"], cwd=HERE, capture_output=True, text=True)
    return dict(commit=cmd.stdout.strip() if cmd.returncode == 0 else None,
                rule_sha256=file_set_digest(CORE[:2]),decision_sha256=file_set_digest(CORE[2:]),
                generator_sha256=file_set_digest(GENERATOR),python=platform.python_version(),
                numpy=numpy.__version__,pyarrow=pyarrow.__version__)

@contextmanager
def exclusive(root):
    """Kernel releases the lock after an interruption, including a killed parent."""
    handle = (root / "checkpoints/writer.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("another generator is writing this dataset") from exc
    try:
        yield
    finally:
        handle.close()

def _config(root, plan, sampling, trajectory, probability):
    if trajectory not in ("none","sampled","all") or not 0 <= probability <= 1:
        raise ValueError("invalid trajectory mode or probability")
    config = dict(schema_version=SCHEMA_VERSION, plan=plan.as_dict(),sampling=asdict(sampling),
                  trajectory=dict(mode=trajectory,probability=probability),software=software())
    config = json.loads(json_text(config))
    path = root / "config.json"
    if path.exists():
        original = json.loads(path.read_text())
        if plan.max_attempts_per_slot < original["plan"]["max_attempts_per_slot"]:
            raise ValueError("the attempt ceiling may only be increased on resume")
        def identity(item):
            plan_values = dict(item["plan"])
            plan_values.pop("max_attempts_per_slot",None)
            return (plan_values,item["sampling"],item["trajectory"],item["schema_version"],
                    *(item["software"][key] for key in ("rule_sha256","decision_sha256","generator_sha256",
                                                       "python","numpy")))
        if identity(original) != identity(config):
            raise ValueError("dataset identity differs from checkpoint; use the original configuration or another directory")
        config["software"] = original["software"]
        config["created_utc"] = original["created_utc"]
    else:
        config["created_utc"] = timestamp()
    write_json(path, config)
    return config

def record_this(spec, plan, trajectory):
    mode = trajectory["mode"]
    if mode != "sampled":
        return mode == "all"
    # Archive all attempts of the first slot in each primary/evaluation category,
    # so reaching that slot's target also archives a completed example.
    if spec.slot_id in _RESERVED:
        return True
    from random import Random
    return Random(spec.trajectory_seed).random() < trajectory["probability"]

_RESERVED = set()

def _worker(spec, plan, sampling, record):
    start = time.monotonic()
    result = simulate(spec, plan, sampling, record)
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform=="darwin" else 1024)
    except ImportError:
        peak = None
    return result, time.monotonic()-start, peak

def family_rss_mib():
    # Include worker processes as well as the coordinator on Linux and macOS.
    if os.name != "posix":
        return None
    try:
        raw = subprocess.check_output(["ps","-axo","pid=,ppid=,rss="], text=True, stderr=subprocess.DEVNULL)
    except (OSError,subprocess.CalledProcessError):
        return None
    rows = [tuple(map(int,line.split())) for line in raw.splitlines() if len(line.split())==3]
    family = {os.getpid()}
    for _ in range(4):
        family.update(pid for pid,parent,_ in rows if parent in family)
    return sum(rss for pid,_,rss in rows if pid in family)/1024

def _runtime(db):
    return {r[0]:r[1] for r in db.execute("SELECT name,value FROM runtime")}

def _manifest(root, db, config, slots):
    split_sizes, drops, checksums, size, serial_seconds = export(root, db, config)
    checksums["config.json"] = digest(root/"config.json")
    size += (root/"config.json").stat().st_size
    attempted, completed, work, worker_peak = db.execute(
        "SELECT COUNT(*),COUNT(winner),SUM(elapsed),MAX(peak_mib) FROM games").fetchone()
    split_games = {k:{"attempted":0,"completed":0} for k in SPLITS}
    for k,n,w in db.execute("SELECT split,COUNT(*),COUNT(winner) FROM games GROUP BY split"):
        split_games[k] = dict(attempted=n,completed=w)
    total = sum(split_sizes.values())
    runtime = _runtime(db)
    generation = runtime.get("generation_seconds",0)
    by_category = Counter(s.category for s in slots if s.split != "heldout_evaluation")
    try:
        import resource
        parent_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform=="darwin" else 1024)
    except ImportError:
        parent_peak = None
    family = runtime.get("family_peak")
    measured_peaks = [v for v in (worker_peak,parent_peak) if v is not None]
    peak_memory = family if family else max(measured_peaks,default=None)
    scope = ("Peak periodically sampled resident memory of coordinator and descendants." if family else
             "Largest measured process high-water mark among the coordinator and workers. Combined concurrent RSS was unavailable.")
    measure = dict(generation_seconds=generation,simulation_worker_seconds=work or 0,
                   attempts_per_second=attempted/generation if generation else 0,
                   completed_per_second=completed/generation if generation else 0,
                   positions_per_second=total/generation if generation else 0,
                   serialization_seconds=serial_seconds,serialization_bytes=size,
                   peak_memory_mib=peak_memory,worker_peak_mib=worker_peak,parent_peak_mib=parent_peak,storage_bytes=size,
                   memory_scope=scope)
    manifest = dict(schema_version=SCHEMA_VERSION, dataset_version="synthetic_v1",created_utc=config["created_utc"],
                    exported_utc=timestamp(),config=config,attempted_games=attempted,completed_games=completed,
                    unfinished_games=attempted-completed,sampled_positions=total,
                    primary_completed=sum(split_games[k]["completed"] for k in SPLITS[:3]),
                    heldout_completed=split_games[SPLITS[-1]]["completed"],allocation_targets=dict(by_category),
                    heldout_target=sum(s.split=="heldout_evaluation" for s in slots),split_sizes=split_sizes,
                    split_games=split_games,cross_split_drops=drops,binary_rows_before_filter=total+sum(drops.values()),
                    checksums=checksums,performance=measure)
    write_json(root/"manifest.json", manifest)
    return manifest

def generate(root, plan, sampling, *, workers=4, trajectory="sampled", probability=0.02,
             attempt_budget=None, progress_seconds=30, run_audit=True):
    if workers<1 or (attempt_budget is not None and attempt_budget<0):
        raise ValueError("invalid worker count or attempt budget")
    root = Path(root)
    (root/"checkpoints").mkdir(parents=True,exist_ok=True)
    (root/"reports").mkdir(exist_ok=True)
    with exclusive(root):
        if (root/"manifest.json").exists() and not (root/"checkpoints/run.sqlite3").is_file():
            raise ValueError("this portable dataset has no resume checkpoint; generate in a fresh output directory")
        config = _config(root,plan,sampling,trajectory,probability)
        db = journal(root/"checkpoints/run.sqlite3")
        try:
            slots = list(make_plan(plan))
            _RESERVED.clear()
            first = {}
            for slot in sorted(slots,key=lambda s:s.slot_id):
                first.setdefault((slot.category,slot.split=="heldout_evaluation"), slot.slot_id)
            _RESERVED.update(first.values())
            max_index = plan.max_attempts_per_slot
            progress = {key:(n or 0, bool(won)) for key,n,won in db.execute(
                "SELECT slot_id,MAX(attempt)+1,COUNT(winner) FROM games GROUP BY slot_id")}
            queue = [(progress.get(s.slot_id,(0,False))[0],order,s) for order,s in enumerate(slots)
                          if not progress.get(s.slot_id,(0,False))[1]
                          and progress.get(s.slot_id,(0,False))[0] < max_index]
            heapify(queue)
            done = sum(1 for _,won in progress.values() if won)
            attempts = sum(n for n,_ in progress.values())
            baseline = attempts
            checkpoint_time = last_print = time.monotonic()
            last_rss = float("-inf")
            peak = _runtime(db).get("family_peak",0)
            pending = {}
            def persist_time():
                nonlocal checkpoint_time,last_rss,peak
                now = time.monotonic()
                increment(db,"generation_seconds",now-checkpoint_time)
                checkpoint_time = now
                if now-last_rss >= 30:
                    current_peak = family_rss_mib()
                    if current_peak is not None:
                        peak = max(peak,current_peak)
                        db.execute("INSERT OR REPLACE INTO runtime VALUES('family_peak',?)",(peak,))
                    last_rss=now if current_peak is not None else float("inf")
            if queue and attempt_budget != 0:
                # One outstanding attempt per logical completion slot guarantees correct quotas
                # across worker counts. A blocked slot cannot stall another category.
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    while pending or queue:
                        wave = min((spec.attempt for _,_,spec in pending.values()),default=queue[0][0] if queue else max_index)
                        while queue and queue[0][0] == wave and len(pending)<workers*2:
                            if attempt_budget is not None and attempts-baseline+len(pending)>=attempt_budget:
                                break
                            index,order,slot = heappop(queue)
                            spec = spec_for(slot,index,plan)
                            future = pool.submit(_worker,spec,plan,sampling,record_this(spec,plan,config["trajectory"]))
                            pending[future] = (order,slot,spec)
                        if not pending:
                            break
                        ready,_ = wait(pending,return_when=FIRST_COMPLETED)
                        with db:
                            for future in ready:
                                order,slot,spec = pending.pop(future)
                                result,elapsed,mem = future.result()
                                append_game(db,spec,result,elapsed,mem)
                                attempts+=1
                                if result.winner is not None:
                                    done+=1
                                elif spec.attempt+1<max_index:
                                    heappush(queue,(spec.attempt+1,order,slot))
                            persist_time()
                        if time.monotonic()-last_print>=progress_seconds or done==len(slots):
                            print(f"{done:,}/{len(slots):,} completed slots, {attempts:,} attempts, "
                                  f"{attempts-done:,} unfinished games",flush=True)
                            last_print=time.monotonic()
            manifest = _manifest(root,db,config,slots)
        finally:
            db.close()
        if run_audit:
            from synthetic_audit import audit
            audit(root)
        return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for mode in ("pilot","full"):
        p=commands.add_parser(mode)
        p.add_argument("--output",type=Path,default=HERE/"datasets/synthetic_v1"/mode)
        p.add_argument("--config",type=Path,help="JSON with plan, sampling and trajectory overrides")
        p.add_argument("--games",type=int,help="primary completed-game slots")
        p.add_argument("--heldout-games",type=int)
        p.add_argument("--seed",type=int)
        p.add_argument("--max-turns",type=int)
        p.add_argument("--max-attempts-per-slot",type=int)
        p.add_argument("--positions-per-game",type=int)
        p.add_argument("--attempt-budget",type=int,help="maximum new attempts in this invocation")
        p.add_argument("--workers",type=int,default=max(1,min(8,os.cpu_count() or 1)))
        p.add_argument("--trajectories",choices=("all","sampled","none"))
        p.add_argument("--trajectory-rate",type=float)
    a=commands.add_parser("audit")
    a.add_argument("dataset",type=Path)
    a.add_argument("--verify-seeds",type=int,default=0)
    args=parser.parse_args()
    if args.command=="audit":
        from synthetic_audit import audit
        audit(args.dataset,verify_seeds=args.verify_seeds)
        return
    pilot = args.command=="pilot"
    base = json.loads(args.config.read_text()) if args.config else {}
    if set(base)-{"plan","sampling","trajectory"}:
        parser.error("config supports plan, sampling and trajectory sections")
    plan_args=dict(namespace=args.command,target_games=5000 if pilot else 80000,
                   heldout_games=100 if pilot else 1000,max_turns=DEFAULT_MAX_TURNS)
    plan_args.update(base.get("plan",{}))
    for key,val in (("target_games",args.games),("heldout_games",args.heldout_games),("master_seed",args.seed),
                    ("max_turns",args.max_turns),("max_attempts_per_slot",args.max_attempts_per_slot)):
        if val is not None:
            plan_args[key]=val
    sampling_args=dict(base.get("sampling",{}))
    if args.positions_per_game is not None:
        sampling_args["positions_per_game"]=args.positions_per_game
    archive=dict(mode="all" if pilot else "sampled",probability=.02)
    archive.update(base.get("trajectory",{}))
    result=generate(args.output,PlanConfig(**plan_args),SamplingConfig(**sampling_args),workers=args.workers,
                    trajectory=args.trajectories or archive["mode"],
                    probability=args.trajectory_rate if args.trajectory_rate is not None else archive["probability"],
                    attempt_budget=args.attempt_budget)
    print(f"{result['primary_completed']:,} primary and {result['heldout_completed']:,} held-out games completed. "
          f"{result['sampled_positions']:,} positions. Report: {args.output/'reports/REPORT.md'}")
if __name__=="__main__":
    main()
