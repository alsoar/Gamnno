"""Optional game-labelled position samples in SQLite, using the existing binary move codec."""

from contextlib import closing, contextmanager
from hashlib import blake2b
import json
from pathlib import Path
import sqlite3
from typing import NamedTuple
import zlib

from training import TrainingRecord, deserialize_record, serialize_record
from game_state import STATE_SIZE


class PositionSample(NamedTuple):
    episode: int
    game_id: str
    turn_number: int
    white: str
    brown: str
    acting_agent: str
    seed: int
    example: TrainingRecord
    features: dict


def pack_sample(turn_number, state, roll, turn, features):
    """Only called on a sampled turn. The board stays in the 27-byte wire format."""
    return (turn_number, serialize_record(TrainingRecord(state, tuple(roll), turn.moves)),
            zlib.compress(json.dumps(features, separators=(",", ":"), allow_nan=False).encode(), 1))


class PositionLog:
    """Append each game and its samples atomically. Repeated runs retain separate episodes."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode INTEGER PRIMARY KEY, game_id TEXT NOT NULL, seed TEXT NOT NULL,
                white TEXT NOT NULL, brown TEXT NOT NULL, winner INTEGER, turns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS positions (
                episode INTEGER NOT NULL, turn INTEGER NOT NULL,
                example BLOB NOT NULL, features BLOB NOT NULL,
                PRIMARY KEY (episode, turn), FOREIGN KEY (episode) REFERENCES episodes(episode)
            ) WITHOUT ROWID;
        """)

    def write(self, game):
        with self.db:
            episode = self.db.execute(
                "INSERT INTO episodes(game_id,seed,white,brown,winner,turns) VALUES(?,?,?,?,?,?)",
                (game.game_id, str(game.seed), game.white, game.brown, game.winner, game.turns),
            ).lastrowid
            self.db.executemany(
                "INSERT INTO positions(episode,turn,example,features) VALUES(?,?,?,?)",
                ((episode, number, example, features) for number, example, features in game.samples),
            )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.db.close()


def iter_positions(path):
    """Stream examples in episode order, with eventual winner or None for unfinished games."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for episode, game_id, seed, white, brown, winner, turn, packed, features in db.execute("""
            SELECT e.episode,e.game_id,e.seed,e.white,e.brown,e.winner,p.turn,p.example,p.features
            FROM episodes e JOIN positions p USING(episode) ORDER BY e.episode,p.turn
        """):
            example = deserialize_record(packed)._replace(winner=winner)
            actor = white if example.state.player == 1 else brown
            yield PositionSample(episode, game_id, turn, white, brown, actor, int(seed), example,
                                 json.loads(zlib.decompress(features)))


@contextmanager
def ranked_decision_inputs(path, seed):
    """Index distinct state/roll keys in a temporary SQLite table, without loading features."""
    salt = str(seed).encode("ascii") + b"/"
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA temp_store=FILE")
        db.execute("""CREATE TEMP TABLE decisions (
            input BLOB PRIMARY KEY, actor TEXT NOT NULL, priority BLOB NOT NULL
        ) WITHOUT ROWID""")
        rows = db.execute("""SELECT e.white,e.brown,p.example
            FROM positions p JOIN episodes e USING(episode) ORDER BY p.episode,p.turn""")
        row_count = 0
        for white, brown, packed in rows:
            row_count += 1
            record = deserialize_record(packed)
            actor = white if record.state.player == 1 else brown
            key = packed[:STATE_SIZE] + bytes(sorted(record.dice))
            rank = blake2b(salt + key, digest_size=8).digest()
            db.execute("INSERT OR IGNORE INTO decisions VALUES(?,?,?)", (key, actor, rank))
        db.execute("CREATE INDEX by_actor_rank ON decisions(actor,priority,input)")
        sizes = dict(db.execute("SELECT actor,COUNT(*) FROM decisions GROUP BY actor"))
        groups = {actor: db.execute("SELECT input FROM decisions WHERE actor=? ORDER BY priority,input",
                                    (actor,)) for actor in sizes}
        yield row_count, sizes, groups
