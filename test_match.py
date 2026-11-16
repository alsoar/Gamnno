"""Match controller tests using real engine moves, dice and the baseline policy."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import UUID

import backgammon as game
from match import Match
from visualizer import _native_color, _native_move


def match_at(points, bars, dice, human_color=game.WHITE):
    state = game.State([points.get(i, 0) for i in range(24)], *bars, game.WHITE)
    with patch.object(game, "starting_state", return_value=state), \
         patch.object(game, "opening_roll", return_value=(game.WHITE, dice)):
        return Match.new(human_color=human_color, seed=11)


class MatchTests(unittest.TestCase):
    def test_seeded_game_controls_both_actor_turns_and_matches_the_archived_game(self):
        archive = json.loads(Path(__file__).with_name("replay-2095.json").read_text())
        for human in (game.WHITE, game.BROWN):
            with self.subTest(human=human):
                match = Match.new(human_color=human, seed=archive["seed"])
                with self.assertRaises(ValueError):
                    match.recording()
                for number, saved in enumerate(archive["turns"], 1):
                    with self.subTest(turn=number):
                        self.assertEqual(match.dice, tuple(saved["roll"]))
                        self.assertEqual(match.state.player, _native_color(saved["player"]))
                        if match.state.player == human:
                            for step in saved["moves"]:
                                match.choose(_native_move(step))
                            if match.winner is None:
                                self.assertTrue(match.preview()[2])
                                match.finish_human_turn()
                        else:
                            match.play_bot_turn()
                        self.assertEqual(len(match.turns), number)
                        self.assertEqual(match.turns[-1], saved)
                self.assertEqual(match.winner, game.WHITE)
                replay = match.recording()
                self.assertEqual({key: replay[key] for key in archive}, archive)
                others = {"white": "human" if human == game.WHITE else "baseline",
                          "brown": "human" if human == game.BROWN else "baseline"}
                self.assertEqual(replay["players"], others)
                self.assertEqual(str(UUID(replay["id"])), match.id)
                self.assertEqual(datetime.fromisoformat(replay["completed_at"]).utcoffset(), timedelta())
                self.assertEqual(json.loads(json.dumps(replay)), replay)
                replay["turns"].clear()
                self.assertEqual(match.recording()["turns"], archive["turns"])

    def test_undo_does_not_commit_moves_frames_or_consume_a_new_roll(self):
        match = match_at({8: 1, 12: 14, 6: -1, 2: -1}, (0, 13), (2, 3))
        initial = match.state
        hit = game.Move(game.BROWN, 6, 8, 2)
        entry = game.Move(game.WHITE, game.BAR, 2, 3)
        match.choose(hit)
        mid, options, done = match.preview()
        self.assertEqual((mid.white_bar, options, done), (1, (entry,), False))
        match.undo()
        self.assertEqual(match.state, initial)
        self.assertEqual(match.preview()[0], initial)
        self.assertEqual((match.dice, match.prefix, match.turns), ((2, 3), (), []))
        match.choose(hit)
        with self.assertRaises(ValueError):
            match.finish_human_turn()
        with self.assertRaises(ValueError):
            match.choose(game.Move(game.WHITE, 12, 15, 3))
        self.assertEqual((match.prefix, len(match.turns)), ((hit,), 0))
        match.choose(entry)
        self.assertEqual(match.preview()[1:], ((), True))
        match.undo()
        self.assertEqual(match.preview()[0], mid)
        match.choose(entry)
        match.finish_human_turn()
        self.assertEqual(match.state, game.apply_turn(initial, (2, 3), (hit, entry)))
        self.assertEqual(len(match.turns), 1)
        self.assertEqual([b[0] for b in match.turns[0]["boards"]], [[1, 13], [0, 14]])

    def test_both_human_and_baseline_can_pass_when_entry_is_blocked(self):
        for human in (game.WHITE, game.BROWN):
            with self.subTest(human=human):
                match = match_at({8: 14, 0: -2, 2: -2, 10: -11}, (1, 0), (1, 3), human)
                start = match.state
                self.assertEqual(match.preview(), (start, (), True))
                if human == game.WHITE:
                    match.finish_human_turn()
                else:
                    match.play_bot_turn()
                self.assertEqual(match.turns, [{"player": 0, "roll": [1, 3], "moves": [], "boards": []}])
                self.assertEqual(match.state, game.apply_turn(start, (1, 3), ()))

    def test_bearoff_wins_on_first_die_and_is_committed_immediately(self):
        match = match_at({12: 14, 22: 1}, (0, 15), (2, 4))
        match.choose(game.Move(game.WHITE, 22, game.OFF, 2))
        self.assertEqual(match.winner, game.WHITE)
        self.assertEqual(match.preview(), (match.state, (), True))
        self.assertEqual(len(match.turns), 1)
        self.assertEqual(match.turns[0]["boards"][-1][26], [1, 0])
        saved = match.recording()
        for action in (match.finish_human_turn, match.undo, match.play_bot_turn):
            with self.subTest(action=action), self.assertRaises(ValueError):
                action()
        self.assertEqual(match.recording(), saved)

    def test_wrong_actor_and_incomplete_turn_do_not_modify_match(self):
        match = match_at({4: 15}, (0, 15), (1, 2), game.BROWN)
        original = match.state
        for action in (lambda: match.choose(game.Move(game.WHITE, 4, 5, 1)),
                       match.finish_human_turn, match.undo):
            with self.subTest(action=action), self.assertRaises(ValueError):
                action()
        self.assertEqual((match.state, match.turns), (original, []))
        match.play_bot_turn()
        after = match.state
        with self.assertRaises(ValueError):
            match.play_bot_turn()
        with self.assertRaises(ValueError):
            match.finish_human_turn()
        self.assertEqual(match.state, after)
        self.assertEqual(len(match.turns), 1)

    def test_unseeded_game_exposes_its_seed_and_unique_session_id(self):
        first = Match.new()
        copied = Match.new(seed=first.seed)
        self.assertEqual((copied.state, copied.dice), (first.state, first.dice))
        self.assertNotEqual(copied.id, first.id)
        with self.assertRaises(ValueError):
            Match.new(human_color=0)


if __name__ == "__main__":
    unittest.main()
