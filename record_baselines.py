"""Rebuild the move histories for the 100 completed games in the revised experiment."""

import csv
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import Random

import backgammon as game
import heuristic
from visualizer import _browser_color, _browser_move


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "baseline_games"


def record(row):
    seed = int(row["seed"])
    rng = Random(seed)
    player, dice = game.opening_roll(rng)
    state = game.starting_state(player)
    opening = game.to_legacy(state)
    turns = []
    totals = dict(self_hits=0, captures=0, checker_moves=0, backward_moves=0,
                  passes=0, acting_blots_sum=0)
    for _ in range(int(row["player_turns_to_win"])):
        selected = heuristic.choose_turn(state, dice)
        metric = heuristic.evaluate_turn(state, selected)
        totals["self_hits"] += metric.own_checkers_hit
        totals["captures"] += metric.captures
        totals["checker_moves"] += len(selected.moves)
        totals["backward_moves"] += metric.backward_moves
        totals["passes"] += int(not selected.moves)
        totals["acting_blots_sum"] += metric.exposed_after
        boards = []
        for i in range(1, len(selected.moves) + 1):
            after, _, _ = game.turn_options(state, dice, selected.moves[:i])
            boards.append(game.to_legacy(after))
        turns.append(dict(player=_browser_color(state.player), roll=dice,
                          moves=list(map(_browser_move, selected.moves)), boards=boards))
        state = selected.state
        if game.winner(state) is not None:
            break
        dice = game.roll_dice(rng)
    winner = game.winner(state)
    expected_winner = game.WHITE if row["winner"] == "White" else game.BROWN
    if (winner != expected_winner or len(turns) != int(row["player_turns_to_win"])
            or any(value != int(row[key]) for key, value in totals.items())):
        raise ValueError(f"seed {seed} differs from the original experiment")
    recording = dict(id=f"baseline-{seed}", players={"white": "baseline", "brown": "baseline"},
                     seed=seed, completed_at=None, opening=opening, turns=turns,
                     winner=_browser_color(winner))
    dest = ARCHIVE / f"seed-{seed}.json.gz"
    data = json.dumps(recording, separators=(",", ":")).encode()
    temp = dest.with_suffix(".tmp")
    temp.write_bytes(gzip.compress(data, mtime=0))
    temp.replace(dest)
    return seed


if __name__ == "__main__":
    ARCHIVE.mkdir(exist_ok=True)
    with (ROOT / "heuristic-revised-100-games.csv").open() as stream:
        rows = sorted(csv.DictReader(stream), key=lambda x: -int(x["player_turns_to_win"]))
    with ProcessPoolExecutor(max_workers=5) as workers:
        list(workers.map(record, rows, chunksize=1))
    print(f"Verified and recorded {len(rows)} baseline games.")
