"""Exercise the native controls in a real Streamlit session."""

import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import backgammon as game
import heuristic
from match import Match


HARNESS = """
import streamlit as st
from play_tab import render_play

class Store:
    def save(self, record):
        st.session_state.setdefault("test_records", {})[record["id"]] = record

render_play(Store())
"""


def open_match(match):
    app = AppTest.from_string(HARNESS, default_timeout=8)
    app.session_state["play_match"] = match
    return app.run()


def click_move(app, move):
    app.selectbox(key="play_source").select((move.checker, move.source)).run()
    key = f"play_move_{move.checker}_{move.source}_{move.target}_{move.die}"
    return app.button(key=key).click().run()


def advance_to_last_turn(human):
    # This recorded game reaches a White bear-off on turn 13.
    match = Match.new(human_color=human, seed=2095)
    # The UI fixture starts at the last turn of the archived original-baseline game.
    with patch.object(match.agent, "choose_turn", side_effect=heuristic.choose_turn):
        for _ in range(12):
            if match.state.player == human:
                turn = heuristic.choose_turn(match.state, match.dice)
                for move in turn.moves:
                    match.choose(move)
                match.finish_human_turn()
            else:
                match.play_bot_turn()
    return match


class PlayTabTests(unittest.TestCase):
    def assert_healthy(self, app):
        self.assertFalse(list(app.exception), list(app.exception))

    def test_native_moves_undo_end_turn_and_visible_bot_response(self):
        match = Match.new(human_color=game.WHITE, seed=2095)
        app = open_match(match)
        self.assert_healthy(app)
        path = heuristic.choose_turn(match.state, match.dice).moves
        click_move(app, path[0])
        self.assertEqual(match.prefix, path[:1])
        app.button(key="play_undo").click().run()
        self.assertEqual(match.prefix, ())
        self.assertTrue(app.button(key="play_end").disabled)
        for move in path:
            click_move(app, move)
        self.assertFalse(app.button(key="play_end").disabled)
        app.button(key="play_end").click().run()
        self.assert_healthy(app)
        self.assertEqual([t["player"] for t in match.turns], [0, 1])
        self.assertEqual(match.prefix, ())
        self.assertEqual(match.state.player, game.WHITE)
        self.assertTrue(any("Baseline’s last turn" in e.label for e in app.expander))

    def test_bar_pass_and_bot_opening(self):
        match = Match.new(human_color=game.WHITE, seed=2095)
        points = [0] * 24
        points[5], points[0], points[2], points[23] = 14, -2, -2, -11
        match.state = game.State(points, 1, 0, game.WHITE)
        match.dice = (1, 3)
        app = open_match(match)
        self.assertEqual(app.button(key="play_end").label, "Pass turn")
        app.button(key="play_end").click().run()
        self.assert_healthy(app)
        self.assertEqual(match.turns[0]["moves"], [])
        self.assertEqual(len(match.turns), 2)
        other = Match.new(human_color=game.BROWN, seed=2095)
        separate = open_match(other)
        self.assert_healthy(separate)
        self.assertEqual(other.state.player, game.BROWN)
        self.assertEqual(len(other.turns), 1)
        self.assertEqual(len(match.turns), 2)

    def test_win_save_and_new_game_as_brown(self):
        match = advance_to_last_turn(game.WHITE)
        moves = heuristic.choose_turn(match.state, match.dice).moves
        for move in moves[:-1]:
            match.choose(move)
        app = open_match(match)
        click_move(app, moves[-1])
        self.assert_healthy(app)
        self.assertEqual(match.winner, game.WHITE)
        self.assertTrue(any("You won" in m.value for m in app.success))
        self.assertNotIn("test_records", app.session_state)
        app.button(key="play_save").click().run()
        self.assertEqual(app.session_state["test_records"][match.id], match.recording())
        self.assertEqual(app.session_state["history_game_type"], "Human vs bot")
        self.assertEqual(app.session_state["history_game"], match.id)
        self.assertTrue(any("saved to History" in m.value for m in app.success))
        app.radio(key="play_new_color").set_value("Brown").run()
        app.selectbox(key="play_opponent").select("hoarder").run()
        app.button(key="play_new").click().run()
        self.assert_healthy(app)
        new = app.session_state["play_match"]
        self.assertNotEqual(new.id, match.id)
        self.assertEqual(new.human_color, game.BROWN)
        self.assertEqual(new.agent.name, "hoarder")
        self.assertEqual(new.state.player, game.BROWN)
        self.assertEqual(len(app.session_state["test_records"]), 1)

    def test_bot_win_can_be_saved(self):
        match = advance_to_last_turn(game.BROWN)
        app = open_match(match)
        self.assert_healthy(app)
        self.assertTrue(any("Baseline won" in m.value for m in app.success))
        app.button(key="play_save").click().run()
        record = app.session_state["test_records"][match.id]
        self.assertEqual(record["winner"], 0)
        self.assertEqual(record["players"], {"white": "baseline", "brown": "human"})


if __name__ == "__main__":
    unittest.main()
