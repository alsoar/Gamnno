"""Check the browser API against the engine without opening a network port."""

import unittest
from unittest.mock import patch

import backgammon as game
import heuristic
from agents import AGENT_NAMES, create_agent
from visualizer import _native_move, game_response


class VisualizerTests(unittest.TestCase):
    def setUp(self):
        self.board = game.to_legacy(game.starting_state())
        self.payload = {"board": self.board, "player": 0, "dice": [3, 1]}

    def test_new_game_uses_both_opening_dice_and_correct_player(self):
        with patch.object(game, "opening_roll", return_value=(game.BROWN, (1, 6))):
            state = game_response("/api/new", {})
        self.assertEqual(state["board"], self.board)
        self.assertEqual(state["player"], 1)
        self.assertEqual(tuple(state["dice"]), (1, 6))
        self.assertIsNone(state["winner"])
        turns = game.legal_turns(game.starting_state(game.BROWN), (1, 6))
        self.assertEqual([item["board"] for item in state["turns"]], [game.to_legacy(t.state) for t in turns])

    def test_roll_and_turn_return_the_same_state_for_the_same_dice(self):
        player_and_board = {"board": self.board, "player": 0}
        with patch.object(game, "roll_dice", return_value=(3, 1)):
            rolled = game_response("/api/roll", player_and_board)
        supplied = game_response("/api/turn", self.payload)
        self.assertEqual(tuple(supplied["dice"]), rolled["dice"])
        self.assertEqual(supplied["turns"], rolled["turns"])

    def test_clicks_follow_a_legal_turn_and_reach_its_advertised_board(self):
        turn = game_response("/api/turn", self.payload)["turns"][0]
        prefix = []
        for step in turn["moves"]:
            response = game_response("/api/choices", dict(self.payload, prefix=prefix))
            self.assertFalse(response["done"])
            self.assertIn(step, response["options"])
            prefix.append(step)
        end = game_response("/api/choices", dict(self.payload, prefix=prefix))
        self.assertTrue(end["done"])
        self.assertEqual(end["options"], [])
        self.assertEqual(end["board"], turn["board"])

    def test_heuristic_endpoint_returns_the_actual_policy_for_both_colors(self):
        for browser_color, color in ((0, game.WHITE), (1, game.BROWN)):
            with self.subTest(color=color):
                payload = dict(self.payload, player=browser_color)
                expected = heuristic.choose_turn(game.starting_state(color), payload["dice"])
                selected = game_response("/api/heuristic", payload)
                self.assertEqual(tuple(map(_native_move, selected["moves"])), expected.moves)
                self.assertEqual(selected["board"], game.to_legacy(expected.state))

    def test_named_agents_can_finish_a_partly_played_roll(self):
        prefix = game_response("/api/choices", self.payload)["options"][:1]
        native = tuple(map(_native_move, prefix))
        start = game.starting_state()
        for name in AGENT_NAMES:
            with self.subTest(name=name):
                result = game_response("/api/agent", dict(self.payload, agent=name, prefix=prefix))
                moves = tuple(map(_native_move, result["moves"]))
                self.assertEqual(moves[:1], native)
                self.assertEqual(result["board"], game.to_legacy(game.apply_turn(start, self.payload["dice"], moves)))
                self.assertTrue(result["explanation"])
                if name != "random":
                    expected = create_agent(name).choose_turn(start, self.payload["dice"], prefix=native)
                    self.assertEqual(moves, expected.moves)

    def test_heuristic_can_take_over_a_partly_played_human_turn(self):
        initial = game_response("/api/heuristic", self.payload)
        options = game_response("/api/choices", self.payload)["options"]
        step = next(move for move in options if move != initial["moves"][0])
        selected = game_response("/api/heuristic", dict(self.payload, prefix=[step]))
        self.assertEqual(selected["moves"][0], step)
        expected = heuristic.choose_turn(game.starting_state(), self.payload["dice"], prefix=[_native_move(step)])
        self.assertEqual(tuple(map(_native_move, selected["moves"])), expected.moves)
        self.assertEqual(selected["board"], game.to_legacy(expected.state))

    def test_each_color_bears_off_only_into_its_own_web_tray(self):
        for checker, src, tray in ((0, 25, 26), (1, 2, 1)):
            with self.subTest(checker=checker):
                board = [(0, 0)] * 27
                board[2], board[25] = (0, 15), (15, 0)
                start = {"board": board, "player": checker, "dice": [1, 3]}
                move = {"checker": checker, "source": src, "target": tray, "die": 1}
                choices = game_response("/api/choices", start)
                self.assertIn(move, choices["options"])
                result = game_response("/api/choices", dict(start, prefix=[move]))
                self.assertTrue(result["done"])
                self.assertEqual(result["board"][tray][checker], 1)
                terminal = game_response("/api/turn", dict(start, board=result["board"]))
                self.assertEqual(terminal["winner"], checker)
                self.assertEqual(terminal["turns"], [])
                move["target"] = 27 - tray
                with self.assertRaises(ValueError):
                    game_response("/api/choices", dict(start, prefix=[move]))

    def test_self_hit_and_mandatory_reentry_keep_browser_coordinates(self):
        board = [(0, 0)] * 27
        board[6], board[8], board[12] = (0, 15), (1, 0), (14, 0)
        hit = {"checker": 1, "source": 6, "target": 8, "die": 2}
        enter = {"checker": 0, "source": 0, "target": 4, "die": 3}
        request = {"board": board, "player": 0, "dice": [2, 3]}
        mid = game_response("/api/choices", dict(request, prefix=[hit]))
        self.assertEqual(mid["board"][0], (1, 0))
        self.assertEqual(mid["options"], [enter])
        end = game_response("/api/choices", dict(request, prefix=[hit, enter]))
        self.assertTrue(end["done"])
        self.assertEqual(end["board"][0], (0, 0))
        self.assertEqual(end["board"][4], (1, 0))

    def test_bad_requests_are_rejected(self):
        cases = (
            ("/api/new", []),
            ("/api/turn", {"dice": [1, 2]}),
            ("/api/turn", dict(self.payload, dice=None)),
            ("/api/turn", dict(self.payload, player=True)),
            ("/api/turn", dict(self.payload, board=[(0, 0)])),
            ("/api/choices", dict(self.payload, prefix=["bad move"])),
            ("/api/choices", dict(self.payload, prefix=[{"die": 3}])),
            ("/api/choices", dict(self.payload, prefix=[{"checker": 0, "source": 0, "target": 25, "die": 3}])),
            ("/api/choices", dict(self.payload, prefix=[{"checker": 0, "source": 26, "target": 3, "die": 3}])),
            ("/api/choices", dict(self.payload, prefix=[{"checker": 0, "source": 2, "target": 0, "die": 3}])),
            ("/api/choices", dict(self.payload, prefix=[{"checker": -1, "source": 2, "target": 3, "die": 1}])),
            ("/api/choices", dict(self.payload, prefix=[{"checker": 0, "source": False, "target": 3, "die": 1}])),
        )
        for path, payload in cases:
            with self.subTest(path=path, payload=payload), self.assertRaises(ValueError):
                game_response(path, payload)


if __name__ == "__main__":
    unittest.main()
