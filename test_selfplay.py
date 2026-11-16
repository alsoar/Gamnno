"""Self-play accounting, seed isolation and turn-limit boundaries."""

import random
import unittest
from unittest.mock import patch

import backgammon as game
import selfplay as sp


def position(stacks, white_bar=0, brown_bar=0, player=game.WHITE):
    points = [stacks.get(i, 0) for i in range(24)]
    return game.State(points, white_bar, brown_bar, player)


def first_turn(state, dice):
    return game.legal_turns(state, dice)[0]


def play_fixture(state, dice, moves):
    def select(current, roll):
        return game.Turn(moves, game.apply_turn(current, roll, moves))

    with patch.object(game, "starting_state", return_value=state), \
         patch.object(game, "opening_roll", return_value=(state.player, dice)):
        return sp.play_game(select, select, seed=5, max_turns=1)


class GameTests(unittest.TestCase):
    def test_omitting_the_seed_reproduces_the_default_opening_and_game(self):
        implicit = sp.play_game(first_turn, first_turn, max_turns=3)
        explicit = sp.play_game(first_turn, first_turn, seed=sp.DEFAULT_SEED, max_turns=3)
        self.assertEqual(implicit, explicit)

    def test_game_stops_on_a_win_at_the_last_allowed_turn(self):
        start = position({0: -15, 23: 15})
        result = play_fixture(start, (1, 6), (game.Move(game.WHITE, 23, game.OFF, 1),))
        self.assertEqual((result.winner, result.turns, result.checker_moves), (game.WHITE, 1, 1))
        self.assertEqual(game.winner(result.final_state), game.WHITE)
        self.assertEqual(result.final_state.player, game.BROWN)
        self.assertIsNone(game.winner(start))

    def test_self_hit_and_reentry_count_one_backward_step_and_no_capture(self):
        start = position({4: -15, 6: 1, 10: 14})
        moves = (game.Move(game.BROWN, 4, 6, 2), game.Move(game.WHITE, game.BAR, 2, 3))
        result = play_fixture(start, (2, 3), moves)
        self.assertIsNone(result.winner)
        self.assertEqual((result.self_hits, result.captures, result.backward_moves), (1, 0, 1))
        self.assertEqual((result.exposed_sum, result.checker_moves, result.passes), (1, 2, 0))
        self.assertEqual(result.final_state.white_bar, 0)

    def test_capture_and_exposure_are_counted_from_the_acting_side(self):
        start = position({0: 15, 2: -1, 11: -14})
        moves = (game.Move(game.WHITE, 0, 2, 2), game.Move(game.WHITE, 2, 3, 1))
        result = play_fixture(start, (2, 1), moves)
        self.assertEqual((result.captures, result.self_hits, result.backward_moves), (1, 0, 0))
        self.assertEqual(result.exposed_sum, 1)
        self.assertEqual(result.final_state.brown_bar, 1)

    def test_pass_is_a_turn_and_contributes_blots_without_a_checker_move(self):
        start = position({0: -2, 2: -2, 10: -11, 8: 13, 9: 1}, white_bar=1)
        result = play_fixture(start, (1, 3), ())
        self.assertIsNone(result.winner)
        self.assertEqual((result.passes, result.turns, result.checker_moves, result.exposed_sum), (1, 1, 0, 1))
        self.assertEqual(result.final_state.player, game.BROWN)

    def test_seed_is_local_and_agent_dispatch_follows_the_opening_roll(self):
        seen = []

        class Brown:
            def choose_turn(self, state, dice):
                seen.append(("brown", int(state.player), dice))
                return first_turn(state, dice)

        def white(state, dice):
            seen.append(("white", int(state.player), dice))
            return first_turn(state, dice)

        global_rng = random.getstate()
        result = sp.play_game(white, Brown(), seed=42, max_turns=3)
        self.assertEqual(seen, [("white", game.WHITE, (6, 1)),
                                ("brown", game.BROWN, (1, 6)),
                                ("white", game.WHITE, (3, 2))])
        self.assertEqual(result, sp.play_game(white, Brown(), seed=42, max_turns=3))
        self.assertEqual(random.getstate(), global_rng)


class SimulationTests(unittest.TestCase):
    def test_aggregate_uses_per_game_per_turn_and_per_move_denominators(self):
        state = game.starting_state()
        results = [
            sp.GameResult(game.WHITE, 3, 0, 6, 2, 0, 2, 6, state),
            sp.GameResult(game.BROWN, 5, 1, 5, 1, 2, 1, 3, state),
            sp.GameResult(None, 6, 3, 3, 0, 1, 1, 5, state),
        ]
        with patch.object(sp, "play_game", side_effect=results) as play:
            report = sp.simulate_games(3, first_turn, first_turn, seed=42, max_turns=6)
        self.assertEqual([call.args[2:] for call in play.call_args_list], [(42, 6), (43, 6), (44, 6)])
        self.assertEqual((report["completed"], report["unfinished"], report["white_wins"], report["brown_wins"]),
                         (2, 1, 1, 1))
        self.assertEqual(report["win_rate_all_starts"], {"white": 1/3, "brown": 1/3})
        self.assertEqual(report["win_rate_completed"], {"white": .5, "brown": .5})
        lengths = report["length_in_player_turns"]
        self.assertEqual(lengths["completed"], {"mean": 4, "median": 4, "histogram": {3: 1, 5: 1}})
        self.assertEqual(lengths["unfinished_observed"], {"mean": 6, "median": 6, "histogram": {6: 1}})
        self.assertEqual(lengths["all_observed"], {"mean": 14/3, "median": 5, "histogram": {3: 1, 5: 1, 6: 1}})
        self.assertEqual((report["turns"], report["passes"], report["checker_moves"]), (14, 4, 14))
        self.assertEqual(report["captures_per_game_started"], 1)
        self.assertEqual(report["self_hits_per_game_started"], 1)
        self.assertEqual(report["acting_blots_after_per_turn"], 1)
        self.assertEqual(report["backward_fraction_of_checker_moves"], 4/14)

    def test_unfinished_pass_has_no_win_or_backward_rate_among_zero_denominators(self):
        only_pass = sp.GameResult(None, 1, 1, 0, 0, 0, 0, 1, game.starting_state())
        with patch.object(sp, "play_game", return_value=only_pass):
            report = sp.simulate_games(1, first_turn, first_turn, max_turns=1)
        self.assertEqual(report["win_rate_all_starts"], {"white": 0, "brown": 0})
        self.assertEqual(report["win_rate_completed"], {"white": None, "brown": None})
        self.assertIsNone(report["length_in_player_turns"]["completed"]["median"])
        self.assertIsNone(report["backward_fraction_of_checker_moves"])
        self.assertEqual(report["acting_blots_after_per_turn"], 1)

    def test_empty_batch_and_invalid_sizes(self):
        report = sp.simulate_games(0, first_turn, first_turn)
        self.assertEqual(report["unfinished"], 0)
        self.assertIsNone(report["win_rate_all_starts"]["white"])
        self.assertIsNone(report["self_hits_per_game_started"])
        for games, limit in ((-1, 1), (0, 0), (3, -1), (True, 5)):
            with self.subTest(games=games, limit=limit), self.assertRaises(ValueError):
                sp.simulate_games(games, first_turn, first_turn, max_turns=limit)


if __name__ == "__main__":
    unittest.main()
