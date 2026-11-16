"""Completed game recordings. SQLite holds saved games; packaged baselines are read-only."""

from contextlib import closing
import csv
from datetime import datetime, timedelta
from functools import lru_cache
import gzip
import json
from pathlib import Path
import sqlite3
import zlib

import backgammon as game
from visualizer import _native_color, _native_move


HERE = Path(__file__).resolve().parent


def _game_type(white, brown):
    humans = (white == "human") + (brown == "human")
    return ("bot-bot", "human-bot", "human-human")[humans]


def _summary(game_id, seed, white, brown, winner, completed_at, turns, moves):
    return {
        "id": game_id, "seed": int(seed), "players": {"white": white, "brown": brown},
        "game_type": _game_type(white, brown), "winner": winner, "completed_at": completed_at,
        "turn_count": turns, "move_count": moves,
    }


@lru_cache(maxsize=1)
def _bundled():
    """Load only the small CSV catalog. Associate ids with fixed local filenames."""
    library = {}
    with (HERE / "heuristic-revised-100-games.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            seed = int(row["seed"])
            path = HERE / "baseline_games" / f"seed-{seed}.json.gz"
            if not path.is_file():
                if seed != 2095 or not (HERE / "replay-2095.json").is_file():
                    continue
                path = HERE / "replay-2095.json"
            game_id = f"baseline-{seed}"
            summary = _summary(game_id, seed, "baseline", "baseline",
                               ("White", "Brown").index(row["winner"]), None,
                               int(row["player_turns_to_win"]), int(row["checker_moves"]))
            library[game_id] = (summary, path)
    return library


def _read_bundled(game_id):
    summary, path = _bundled()[game_id]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        record = json.load(file)
    record.update(id=game_id, players=summary["players"].copy(), completed_at=None)
    return record


def _validate(record):
    """Check a new recording once, including all intermediate boards and its final win."""
    try:
        game_id, seed, players = record["id"], record["seed"], record["players"]
        completed_at, winner, turns = record["completed_at"], record["winner"], record["turns"]
        if not isinstance(game_id, str) or not game_id.strip() or type(seed) is not int:
            raise ValueError("a recording needs an id and an integer seed")
        if set(players) != {"white", "brown"} or any(role not in ("human", "baseline") for role in players.values()):
            raise ValueError("choose a human or baseline player for each color")
        if not isinstance(completed_at, str):
            raise ValueError("completed_at must be a UTC timestamp")
        stamp = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        if stamp.utcoffset() != timedelta(0):
            raise ValueError("completed_at must be a UTC timestamp")
        if type(winner) is not int or winner not in (0, 1) or not turns:
            raise ValueError("only completed games with a winner can be saved")
        first_player = _native_color(turns[0]["player"])
        state = game.from_legacy(record["opening"], first_player)
        move_count = 0
        for turn in turns:
            if _native_color(turn["player"]) != state.player:
                raise ValueError("incorrect player in recorded turn")
            moves = tuple(_native_move(move) for move in turn["moves"])
            if len(moves) != len(turn["boards"]):
                raise ValueError("expected one board per checker move")
            end = game.apply_turn(state, turn["roll"], moves)
            scratch = game._scratch(state)
            for move, board in zip(moves, turn["boards"]):
                game._push(scratch, move)
                actual = game.from_legacy(board, state.player)
                if bytes(scratch) != game.state_key(actual):
                    raise ValueError("a recorded frame differs from its move")
            move_count += len(moves)
            state = end
        if game.winner(state) != _native_color(winner):
            raise ValueError("recorded winner does not match the final board")
        return _summary(game_id, seed, players["white"], players["brown"], winner,
                        completed_at, len(turns), move_count)
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError("malformed game recording") from error


def _encode(record):
    return json.dumps(record, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")


class HistoryStore:
    """A file-backed store safe to share across sessions. Every operation closes its connection."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS recordings (
                    id TEXT PRIMARY KEY, seed TEXT NOT NULL, white TEXT NOT NULL, brown TEXT NOT NULL,
                    winner INTEGER NOT NULL, completed_at TEXT, turns INTEGER NOT NULL,
                    moves INTEGER NOT NULL, payload BLOB NOT NULL
                )
            """)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def save(self, record):
        """Return True for a new completed game or False for an identical repeat save."""
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("a game recording needs a string id")
        game_id = record["id"]
        data = _encode(record)
        if game_id in _bundled():
            if data == _encode(_read_bundled(game_id)):
                return False
            raise ValueError("a bundled game already uses this id")
        summary = _validate(record)
        with closing(self._connect()) as db, db:
            result = db.execute("""
                INSERT INTO recordings(id, seed, white, brown, winner, completed_at, turns, moves, payload)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING
            """, (game_id, str(summary["seed"]), summary["players"]["white"], summary["players"]["brown"],
                  summary["winner"], summary["completed_at"], summary["turn_count"],
                  summary["move_count"], zlib.compress(data)))
            if result.rowcount:
                return True
            existing = db.execute("SELECT payload FROM recordings WHERE id = ?", (game_id,)).fetchone()[0]
            if zlib.decompress(existing) == data:
                return False
            raise ValueError("a different completed game already uses this id")

    def list_games(self):
        """Return summaries. Newly saved games come first, then baseline games by seed."""
        with closing(self._connect()) as db:
            rows = db.execute("""
                SELECT id, seed, white, brown, winner, completed_at, turns, moves
                FROM recordings ORDER BY completed_at DESC, rowid DESC
            """)
            saved = [_summary(*row) for row in rows]
        for summary, _ in _bundled().values():
            saved.append(dict(summary, players=summary["players"].copy()))
        return saved

    def load(self, game_id):
        """Load the selected recording, including each move's board. Raise KeyError for a missing id."""
        if game_id in _bundled():
            return _read_bundled(game_id)
        with closing(self._connect()) as db:
            row = db.execute("SELECT payload FROM recordings WHERE id = ?", (game_id,)).fetchone()
        if row is None:
            raise KeyError(game_id)
        return json.loads(zlib.decompress(row[0]))
