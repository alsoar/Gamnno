"""Transactional game journal and portable, compressed synthetic value records."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import tempfile
import time

import pyarrow as pa
import pyarrow.parquet as pq

from agents import AGENT_NAMES
import backgammon as game
from synthetic_game import PHASES, policy_identity

SPLITS = ("train", "validation", "test", "heldout_evaluation")
SCHEMA_VERSION = 1
GAME_COLUMNS = (
    "game_id", "slot_id", "attempt", "split", "category", "subcategory", "generation_seed",
    "white_agent", "brown_agent", "mixture_type", "white_meta", "brown_meta", "opener", "winner",
    "turns", "reason", "initial", "final", "policy_trace", "stats", "elapsed", "peak_mib", "trajectory",
)
JOURNAL = """
CREATE TABLE IF NOT EXISTS games(
    game_id TEXT PRIMARY KEY, slot_id TEXT NOT NULL, attempt INTEGER NOT NULL, split TEXT NOT NULL,
    category TEXT NOT NULL, subcategory TEXT NOT NULL, generation_seed INTEGER NOT NULL UNIQUE,
    white_agent TEXT NOT NULL, brown_agent TEXT NOT NULL, mixture_type TEXT NOT NULL,
    white_meta TEXT NOT NULL, brown_meta TEXT NOT NULL, opener INTEGER NOT NULL, winner INTEGER,
    turns INTEGER NOT NULL, reason TEXT, initial BLOB NOT NULL, final BLOB NOT NULL,
    policy_trace BLOB NOT NULL, stats TEXT NOT NULL, elapsed REAL NOT NULL, peak_mib REAL,
    trajectory BLOB, UNIQUE(slot_id,attempt)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_win_per_slot ON games(slot_id) WHERE winner IS NOT NULL;
CREATE TABLE IF NOT EXISTS samples(game_id TEXT NOT NULL REFERENCES games(game_id),turn_number INTEGER NOT NULL,
    state BLOB NOT NULL,phase INTEGER NOT NULL,policy INTEGER NOT NULL,PRIMARY KEY(game_id,turn_number)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS runtime(name TEXT PRIMARY KEY,value REAL NOT NULL);
"""

def json_text(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)

def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)

def digest(path):
    state = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()

def journal(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(JOURNAL)
    return db

def increment(db, name, value):
    db.execute("INSERT INTO runtime VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value",
               (name, value))

def append_game(db, spec, result, elapsed, peak):
    # Each attempt and every sample form one checkpoint transaction.
    if result.winner != game.winner(game.deserialize_state(result.final)):
        raise ValueError("winner disagrees with final state")
    if (result.winner is None and result.samples) or result.turns != len(result.policy_trace):
        raise ValueError("invalid censored samples or turn count")
    if result.initial != game.state_key(game.starting_state(result.opener)):
        raise ValueError("incorrect initial board")
    white, brown = spec.white.name, spec.brown.name
    mixture = spec.mixture_type
    values = (spec.game_id, spec.slot_id, spec.attempt, spec.split, spec.category, spec.subcategory,
              spec.generation_seed, white, brown, mixture, json_text(result.white_meta),
              json_text(result.brown_meta), result.opener, result.winner, result.turns,
              "turn_limit" if result.winner is None else None, result.initial, result.final,
              result.policy_trace, json_text(result.stats), elapsed, peak, result.trajectory)
    _insert_game(db, values)
    rows = []
    for turn, raw, phase, policy in result.samples:
        state = game.deserialize_state(raw)
        policy_identity(policy)
        if (not 2 <= turn <= result.turns or game.winner(state) is not None
                or int(state.player) != result.opener * (-1 if turn % 2 == 0 else 1)
                or not 0 <= phase < len(PHASES) or policy != result.policy_trace[turn-1]):
            raise ValueError("invalid pre-roll position")
        rows.append((spec.game_id, turn, raw, phase, policy))
    db.executemany("INSERT INTO samples VALUES(?,?,?,?,?)", rows)

def _insert_game(db, values):
    db.execute("INSERT INTO games VALUES(" + ",".join("?" for _ in GAME_COLUMNS) + ")", tuple(values))

VALUE_SCHEMA = pa.schema([
    ("game_id", pa.string()), ("position_id", pa.uint32()), ("turn_number", pa.uint32()),
    ("board", pa.binary(24)), ("white_bar", pa.uint8()), ("brown_bar", pa.uint8()),
    ("player_to_move", pa.int8()), ("winner", pa.int8()), ("win_target", pa.uint8()),
    ("white_agent", pa.uint16()), ("brown_agent", pa.uint16()), ("matchup_category", pa.uint16()),
    ("mixture_type", pa.uint8()), ("game_phase", pa.uint8()), ("game_length", pa.uint32()),
    ("generation_seed", pa.int64()), ("source_policy_metadata", pa.uint8()),
])
GAME_SCHEMA = pa.schema([
    (col, pa.int64() if col == "generation_seed" else pa.uint32() if col in {"attempt", "turns"}
     else pa.int8() if col in {"opener", "winner"} else pa.binary(27) if col in {"initial","final"}
     else pa.binary() if col == "policy_trace" else pa.float64() if col in {"elapsed","peak_mib"}
     else pa.string()) for col in GAME_COLUMNS if col != "trajectory"
] + [("trajectory_file", pa.string()), ("trajectory_offset", pa.uint64()), ("trajectory_size", pa.uint32())])

def _write_part(path, rows, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")

def _dictionary(names):
    return {str(i): value for i, value in enumerate(names)}

def make_codebook(db, config):
    agents = db.execute("SELECT white_agent FROM games UNION SELECT brown_agent FROM games ORDER BY 1")
    def names(column):
        return [r[0] for r in db.execute(f"SELECT DISTINCT {column} FROM games ORDER BY 1")]
    return {"agents": _dictionary(r[0] for r in agents), "categories": _dictionary(names("category")),
            "mixture_types": _dictionary(names("mixture_type")), "strategies": _dictionary(AGENT_NAMES),
            "phases": _dictionary(PHASES), "policy_config_version": config["software"]["decision_sha256"],
            "mixture_weights": config["plan"]["mixture_weights"], "heldout_weights": config["plan"]["heldout_weights"],
            "epsilons": config["plan"]["epsilons"], "exploration_flag": 128}

def _inverse(codes):
    return {name: int(i) for i, name in codes.items()}

_NEGATE = bytes((-n) & 255 for n in range(256))

def normalized_key(raw):
    """Compact model input in physical White's direction, including both bars."""
    return raw if raw[26] == 1 else raw[:24][::-1].translate(_NEGATE) + bytes((raw[25],raw[24],1))

def _export_values(staging, db, codes):
    agents, categories, mixtures = (_inverse(codes[key]) for key in ("agents","categories","mixture_types"))
    counts, drops = {}, {}
    # Disk-backed state ownership avoids collecting a million Python objects.
    with sqlite3.connect(staging / "owners.sqlite") as owners:
        owners.execute("CREATE TABLE seen(state BLOB PRIMARY KEY, split TEXT NOT NULL) WITHOUT ROWID")
        for split in SPLITS:
            rows, part, kept, dropped = [], 0, 0, 0
            sql = """SELECT s.*, g.winner,g.white_agent,g.brown_agent,g.category,g.mixture_type,
                     g.turns,g.generation_seed FROM games g JOIN samples s USING(game_id)
                     WHERE g.split=? ORDER BY g.game_id,s.turn_number"""
            for r in db.execute(sql, (split,)):
                raw = r["state"]
                key = normalized_key(raw)
                owners.execute("INSERT OR IGNORE INTO seen VALUES(?,?)", (key,split))
                if owners.execute("SELECT split FROM seen WHERE state=?", (key,)).fetchone()[0] != split:
                    dropped += 1
                    continue
                player = 1 if raw[26] == 1 else -1
                rows.append(dict(game_id=r["game_id"], position_id=r["turn_number"], turn_number=r["turn_number"],
                                 board=raw[:24], white_bar=raw[24], brown_bar=raw[25], player_to_move=player,
                                 winner=r["winner"], win_target=int(player == r["winner"]),
                                 white_agent=agents[r["white_agent"]], brown_agent=agents[r["brown_agent"]],
                                 matchup_category=categories[r["category"]], mixture_type=mixtures[r["mixture_type"]],
                                 game_phase=r["phase"], game_length=r["turns"],generation_seed=r["generation_seed"],
                                 source_policy_metadata=r["policy"]))
                kept += 1
                if len(rows) == 8192:
                    _write_part(staging / split / f"part-{part:05}.parquet", rows, VALUE_SCHEMA)
                    rows, part = [], part + 1
            if rows or not part:
                _write_part(staging / split / f"part-{part:05}.parquet", rows, VALUE_SCHEMA)
            counts[split], drops[split] = kept, dropped
    return counts, drops

def _export_games(staging, db):
    (staging / "trajectories").mkdir()
    cursor = db.execute("SELECT * FROM games ORDER BY game_id")
    part = 0
    while True:
        input_rows = cursor.fetchmany(128)
        if not input_rows:
            if part == 0:
                _write_part(staging / "games/part-00000.parquet", [], GAME_SCHEMA)
            break
        output_rows = []
        rel = f"trajectories/part-{part:05}.bin"
        with (staging / rel).open("wb") as trajectories:
            for row in input_rows:
                item = dict(row)
                blob = item.pop("trajectory")
                item.update(trajectory_file=rel if blob else None, trajectory_offset=trajectories.tell() if blob else None,
                            trajectory_size=len(blob) if blob else None)
                if blob:
                    trajectories.write(blob)
                output_rows.append(item)
        if not (staging / rel).stat().st_size:
            (staging / rel).unlink()
        _write_part(staging / "games" / f"part-{part:05}.parquet", output_rows, GAME_SCHEMA)
        part += 1

def export(root, db, config):
    start = time.monotonic()
    with tempfile.TemporaryDirectory(dir=root / "checkpoints") as tmp:
        stage = Path(tmp)
        codes = make_codebook(db, config)
        write_json(stage / "agent_metadata.json", codes)
        splits, drops = _export_values(stage, db, codes)
        _export_games(stage, db)
        checksums = {}
        size = 0
        for file in sorted(stage.rglob("*")):
            if not file.is_file() or file.name == "owners.sqlite":
                continue
            rel = file.relative_to(stage)
            checksums[str(rel)] = digest(file)
            size += file.stat().st_size
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            file.replace(target)
        # Remove stale managed shards if a resumed export changed the number of parts.
        for folder in (*SPLITS, "games","trajectories"):
            for file in (root / folder).glob("part-*"):
                if str(file.relative_to(root)) not in checksums:
                    file.unlink()
        return splits, drops, checksums, size, time.monotonic() - start

def _portable_rows(root, folder):
    for file in sorted((root / folder).glob("part-*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=1024):
            yield from batch.to_pylist()

def state_from_row(row):
    return game.deserialize_state(row["board"] + bytes((row["white_bar"],row["brown_bar"],row["player_to_move"] & 255)))

def value_input(row):
    """Only information available before the roll, normalized to the player to move."""
    from training import encode_state
    return encode_state(state_from_row(row))

@contextmanager
def evidence_db(root):
    """Read the original checkpoint or reconstruct evidence from distributed portable shards."""
    root = Path(root)
    path = root / "checkpoints/run.sqlite3"
    if path.is_file():
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()
        return
    with tempfile.TemporaryDirectory() as tmp:
        db = journal(Path(tmp) / "evidence.sqlite")
        try:
            with db:
                for item in _portable_rows(root,"games"):
                    file, offset, size = (item.pop(k) for k in ("trajectory_file","trajectory_offset","trajectory_size"))
                    data = None
                    if file:
                        path = (root / file).resolve()
                        if path.parent != (root / "trajectories").resolve():
                            raise ValueError("invalid trajectory location")
                        with path.open("rb") as stream:
                            stream.seek(offset)
                            data = stream.read(size)
                            if len(data) != size:
                                raise ValueError("truncated trajectory")
                    item["trajectory"] = data
                    _insert_game(db, (item[key] for key in GAME_COLUMNS))
                for split in SPLITS:
                    db.executemany("INSERT INTO samples VALUES(?,?,?,?,?)", (
                        (r["game_id"], r["turn_number"],game.state_key(state_from_row(r)),r["game_phase"],r["source_policy_metadata"])
                        for r in _portable_rows(root, split)))
            yield db
        finally:
            db.close()
