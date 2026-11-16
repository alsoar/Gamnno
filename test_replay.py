"""Verify every frame of the seed 2095 replay against the game and the experiment."""

import csv
import json
import unittest
from pathlib import Path

import backgammon as game
import heuristic
from selfplay import play_game
from visualizer import _browser_color, _browser_move, _native_color, _native_move


HERE = Path(__file__).resolve().parent
REPLAY = HERE / "replay-2095.json"
EXPERIMENT = HERE / "heuristic-revised-100-games.csv"


def browser_board(state):
    return [list(cell) for cell in game.to_legacy(state)]


def record_game(seed):
    """Record both revised heuristic agents through the same runner used for the CSV."""
    history = []

    def record(state, dice):
        turn = heuristic.choose_turn(state, dice)
        frames = []
        for length in range(1, len(turn.moves) + 1):
            preview, _, _ = game.turn_options(state, dice, turn.moves[:length])
            frames.append(browser_board(preview))
        history.append({
            "player": _browser_color(state.player), "roll": list(dice),
            "moves": [_browser_move(move) for move in turn.moves], "boards": frames,
        })
        return turn

    result = play_game(record, record, seed=seed)
    if result.winner is None:
        raise ValueError("the recorded game must end with a winner")
    return {"seed": seed, "opening": browser_board(game.starting_state()),
            "turns": history, "winner": _browser_color(result.winner)}


class ReplayTests(unittest.TestCase):
    def test_every_saved_die_move_and_frame_matches_seeded_selfplay(self):
        replay = json.loads(REPLAY.read_text())
        self.assertEqual(replay["seed"], 2095)
        self.assertEqual(replay, record_game(2095))

    def test_saved_game_is_legal_and_agrees_with_every_recorded_statistic(self):
        replay = json.loads(REPLAY.read_text())
        with EXPERIMENT.open(newline="") as stream:
            row, = (r for r in csv.DictReader(stream) if int(r["seed"]) == replay["seed"])
        turns = replay["turns"]
        self.assertEqual(len(turns), 13)
        self.assertEqual(len(turns), int(row["player_turns_to_win"]))
        state = game.from_legacy(replay["opening"], _native_color(turns[0]["player"]))
        totals = dict(self_hits=0, captures=0, checker_moves=0, backward_moves=0,
                      passes=0, acting_blots_sum=0)
        for number, saved in enumerate(turns, 1):
            with self.subTest(turn=number):
                self.assertIsNone(game.winner(state))
                self.assertEqual(_browser_color(state.player), saved["player"])
                moves = tuple(_native_move(m) for m in saved["moves"])
                end = game.apply_turn(state, saved["roll"], moves)
                self.assertEqual(len(saved["boards"]), len(moves))
                if saved["boards"]:
                    self.assertEqual(saved["boards"][-1], browser_board(end))
                metrics = heuristic.evaluate_turn(state, game.Turn(moves, end))
                totals["self_hits"] += metrics.own_checkers_hit
                totals["captures"] += metrics.captures
                totals["checker_moves"] += len(moves)
                totals["backward_moves"] += metrics.backward_moves
                totals["passes"] += int(not moves)
                totals["acting_blots_sum"] += metrics.exposed_after
                state = end
        self.assertEqual(totals, {name: int(row[name]) for name in totals})
        color = game.winner(state)
        self.assertEqual(color, game.WHITE)
        self.assertEqual(_browser_color(color), replay["winner"])
        self.assertEqual(row["winner"], "White")


if __name__ == "__main__":
    unittest.main()
