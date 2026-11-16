"""Run with python3 -m unittest -v in this directory."""

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call

import numpy as np

from backgammon import (
    BAR, OFF, BROWN, WHITE, Move, State, apply_turn, deserialize_state,
    from_legacy, legal_turns, opening_roll, roll_dice, serialize_state,
    sample_turn, starting_state, state_key, to_legacy, turn_options, winner,
)


def position(points, bars=(0, 0), player=WHITE):
    """Build a full-count position from {point: signed count} and (white, brown) bars."""
    board = [points.get(i, 0) for i in range(24)]
    return State(board, *bars, player)


def mirrored(state):
    return State(-state.board[::-1], state.brown_bar, state.white_bar, -state.player)


def with_next_player(state):
    return State(state.board, state.white_bar, state.brown_bar, -state.player)


class StateTests(unittest.TestCase):
    def test_standard_start_dtypes_and_color_order(self):
        state = starting_state()
        self.assertEqual((int(WHITE), int(BROWN)), (1, -1))
        self.assertEqual(state.player, WHITE)
        self.assertEqual((int(state.white_off), int(state.brown_off)), (0, 0))
        self.assertIsNone(winner(state))
        self.assertEqual(state, position({0: 2, 11: 5, 16: 3, 18: 5,
                                          23: -2, 12: -5, 7: -3, 5: -5}))
        self.assertEqual(state.board.shape, (24,))
        self.assertEqual(state.board.dtype, np.int8)
        self.assertIsInstance(state.white_bar, np.uint8)
        self.assertIsInstance(state.brown_bar, np.uint8)
        self.assertIsInstance(state.player, np.int8)
        self.assertEqual(starting_state(BROWN), mirrored(state))

    def test_freezes_input_and_blocks_writable_views(self):
        raw = np.zeros(24, dtype=np.int8)
        raw[5], raw[9] = 13, -11
        frozen = State(raw, 2, 4, WHITE)
        old_key = state_key(frozen)
        raw[5] = 0
        self.assertEqual(frozen.board[5], 13)
        with self.assertRaises(ValueError):
            frozen.board[5] = 0
        with self.assertRaises(ValueError):
            frozen.board.setflags(write=True)
        view = frozen.board
        view.shape = (4, 6)
        self.assertEqual(frozen.board.shape, (24,))
        with self.assertRaises((AttributeError, TypeError)):
            frozen.player = BROWN
        self.assertEqual(old_key, state_key(frozen))

    def test_serialization_hash_and_side_to_move(self):
        white = position({4: 13, 11: -10}, (2, 5))
        other = with_next_player(white)
        restored = deserialize_state(serialize_state(white))
        self.assertEqual(restored, white)
        self.assertEqual(hash(restored), hash(white))
        self.assertIsInstance(state_key(white), bytes)
        self.assertEqual(state_key(restored), state_key(white))
        self.assertNotEqual(state_key(other), state_key(white))
        self.assertEqual(len({white, other, restored}), 2)

    def test_stable_hash_across_processes_and_hash_seeds(self):
        query = "from backgammon import starting_state; print(hash(starting_state()))"
        for seed in ("72", "613"):
            output = subprocess.check_output(
                [sys.executable, "-B", "-c", query], cwd=Path(__file__).parent,
                env={**os.environ, "PYTHONHASHSEED": seed}, text=True,
            )
            self.assertEqual(int(output), hash(starting_state()))

    def test_wire_fields_and_mutable_buffer_ownership(self):
        state = position({0: -11, 23: 13}, (2, 4), BROWN)
        expected = state.board.tobytes() + bytes((2, 4, 255))
        self.assertEqual(serialize_state(state), expected)
        self.assertEqual(state_key(state), expected)
        buf = bytearray(expected)
        copy = deserialize_state(buf)
        buf[26] = 1
        self.assertEqual(copy, state)
        for invalid in (expected[:-1], expected + b"x", expected[:-1] + b"\0",
                        expected[:24] + bytes((255, 4, 255))):
            with self.subTest(wire=invalid), self.assertRaises(ValueError):
                deserialize_state(invalid)

    def test_terminal_winner_inferred_from_counts_for_each_color(self):
        white = position({23: 14, 0: -15}, player=BROWN)
        brown = mirrored(white)
        for state, color in ((white, WHITE), (brown, BROWN)):
            with self.subTest(color=color):
                self.assertEqual(winner(state), color)
                self.assertEqual(state.white_off + state.brown_off, 1)
                self.assertEqual(deserialize_state(serialize_state(state)), state)
                self.assertEqual(from_legacy(to_legacy(state), state.player), state)
                self.assertEqual(legal_turns(state, (1, 6)), ())
                with self.assertRaises(ValueError):
                    apply_turn(state, (1, 6), ())

    def test_state_rejects_bad_shapes_counts_and_narrowing_overflow(self):
        valid = starting_state().board.tolist()
        for board, white_bar, brown_bar, player in [
            (valid[:-1], 0, 0, WHITE), ([valid], 0, 0, WHITE),
            ([256] + valid[1:], 0, 0, WHITE), ([-129] + valid[1:], 0, 0, WHITE),
            ([1.25] + valid[1:], 0, 0, WHITE), (valid, -1, 0, WHITE),
            (valid, 256, 0, WHITE), (valid, 0, 1, WHITE), (valid, 0, 0, 0),
            ([0] * 24, 0, 0, WHITE), ([0] * 24, 14, 14, WHITE),
        ]:
            with self.subTest(board=board, bars=(white_bar, brown_bar), player=player):
                with self.assertRaises((TypeError, ValueError, OverflowError)):
                    State(board, white_bar, brown_bar, player)

    def test_legacy_geometry_and_shared_bar_roundtrip(self):
        state = position({0: 13, 23: -11}, (2, 4), BROWN)
        expected = [(0, 0)] * 27
        expected[0], expected[2], expected[25] = (2, 4), (13, 0), (0, 11)
        self.assertEqual(to_legacy(state), tuple(expected))
        self.assertEqual(from_legacy(expected, BROWN), state)
        self.assertEqual(to_legacy(position({23: 14, 0: -15}))[26], (1, 0))
        self.assertEqual(to_legacy(position({23: 15, 0: -14}))[1], (0, 1))

    def test_legacy_converter_rejects_impossible_or_incomplete_positions(self):
        base = list(to_legacy(starting_state()))
        bad_cells = [(0, (-1, 0)), (4, (1, 1)), (1, (1, 0)), (26, (0, 1)),
                     (4, (1,)), (2, (1.1, 0)), (2, (1, 0)), (2, (200, 0))]
        for i, value in bad_cells:
            cells = base.copy()
            cells[i] = value
            with self.subTest(i=i, value=value), self.assertRaises((TypeError, ValueError)):
                from_legacy(cells, WHITE)
        with self.assertRaises(ValueError):
            from_legacy(base[:-1], WHITE)


class MovementTests(unittest.TestCase):
    def test_forward_hit_then_backward_move_onto_same_color(self):
        state = position({3: 15, 5: -1, 7: -1, 10: -13})
        moves = (Move(WHITE, 3, 5, 2), Move(BROWN, 7, 10, 3))
        self.assertEqual(apply_turn(state, (2, 3), moves),
                         position({3: 14, 5: 1, 10: -14}, (0, 1), BROWN))

    def test_both_actions_obey_destination_blocking(self):
        state = position({3: 13, 10: 2, 5: -2, 8: -13})
        for illegal in (Move(WHITE, 3, 5, 2), Move(BROWN, 8, 10, 2)):
            with self.subTest(illegal=illegal), self.assertRaises(ValueError):
                apply_turn(state, (2, 1), (illegal, Move(WHITE, 3, 4, 1)))

    def test_self_hit_requires_reentry_on_the_very_next_die(self):
        state = position({6: 1, 10: 14, 4: -15})
        hit = Move(BROWN, 4, 6, 2)
        mid, next_moves, done = turn_options(state, (2, 3), (hit,))
        self.assertFalse(done)
        self.assertEqual(mid.white_bar, 1)
        self.assertEqual(mid.player, WHITE)
        entry = Move(WHITE, BAR, 2, 3)
        self.assertEqual(next_moves, (entry,))
        self.assertEqual(apply_turn(state, (2, 3), (hit, entry)),
                         position({2: 1, 10: 14, 4: -14, 6: -1}, player=BROWN))
        with self.assertRaises(ValueError):
            apply_turn(state, (2, 3), (hit, Move(BROWN, 4, 7, 3)))

    def test_opponent_checker_stays_on_board(self):
        state = position({12: 15, 23: -15})
        turns = legal_turns(state, (1, 3))
        self.assertTrue(turns)
        self.assertTrue(all(m.checker == WHITE for t in turns for m in t.moves))
        with self.assertRaises(ValueError):
            apply_turn(state, (1, 3), (Move(BROWN, 23, OFF, 1),))

    def test_cannot_move_opponent_checker_off_bar(self):
        state = position({8: 15}, (0, 15))
        for t in legal_turns(state, (1, 4)):
            self.assertEqual(t.state.brown_bar, 15)
            self.assertTrue(all(m.checker == WHITE for m in t.moves))


class BarAndDiceTests(unittest.TestCase):
    def test_opening_ties_reroll_and_higher_roller_plays_both_dice(self):
        for dice, first in (((5, 3), WHITE), ((1, 6), BROWN)):
            with self.subTest(dice=dice):
                rng = Mock()
                rng.randint.side_effect = [4, 4, *dice]
                self.assertEqual(opening_roll(rng), (first, dice))
                self.assertEqual(rng.randint.call_args_list, [call(1, 6)] * 4)

    def test_regular_roll_keeps_doubles(self):
        rng = Mock()
        rng.randint.side_effect = [6, 6]
        self.assertEqual(roll_dice(rng), (6, 6))
        self.assertEqual(rng.randint.call_args_list, [call(1, 6)] * 2)

    def test_shared_bar_and_hit_on_reentry(self):
        state = position({10: 14, 1: -1}, (1, 14))
        moves = (Move(WHITE, BAR, 1, 2), Move(WHITE, 10, 14, 4))
        self.assertEqual(apply_turn(state, (2, 4), moves),
                         position({1: 1, 10: 13, 14: 1}, (0, 15), BROWN))

    def test_every_own_checker_on_bar_has_priority(self):
        state = position({10: 13, 18: -15}, (2, 0))
        for t in legal_turns(state, (1, 4)):
            self.assertEqual(len(t.moves), 2)
            self.assertTrue(all((m.checker, m.source) == (WHITE, BAR) for m in t.moves))
            self.assertEqual(t.state.white_bar, 0)

    def test_blocked_bar_is_one_forced_pass_and_switches_player(self):
        state = position({8: 14, 0: -2, 2: -2, 10: -11}, (1, 0))
        turns = legal_turns(state, (1, 3))
        self.assertEqual([t.moves for t in turns], [()])
        self.assertEqual(turns[0].state, with_next_player(state))
        self.assertEqual(apply_turn(state, (1, 3), ()), turns[0].state)
        self.assertEqual(turn_options(state, (1, 3)), (state, (), True))

    def test_a_blocked_die_becomes_available_after_entry(self):
        state = position({8: 14, 0: -2, 17: -13}, (1, 0))
        turns = legal_turns(state, (1, 2))
        self.assertTrue(turns)
        for t in turns:
            self.assertEqual(len(t.moves), 2)
            self.assertEqual(t.moves[0], Move(WHITE, BAR, 1, 2))
        moves = (Move(WHITE, BAR, 1, 2), Move(BROWN, 17, 18, 1))
        self.assertEqual(apply_turn(state, (1, 2), moves),
                         position({1: 1, 8: 14, 0: -2, 17: -12, 18: -1}, player=BROWN))

    def test_short_self_hit_is_illegal_when_another_line_uses_both(self):
        state = position({5: 1, 8: 14, 1: -2, 4: -1}, (0, 12))
        hit = Move(BROWN, 4, 5, 1)
        self.assertTrue(all(len(t.moves) == 2 for t in legal_turns(state, (1, 2))))
        self.assertNotIn(hit, turn_options(state, (1, 2))[1])
        with self.assertRaises(ValueError):
            apply_turn(state, (1, 2), (hit,))
        with self.assertRaises(ValueError):
            turn_options(state, (1, 2), (hit,))

    def test_higher_die_required_if_either_can_be_played_but_only_one_fits(self):
        # White's stack on 23 cannot overshoot while the checker on 17 is farther.
        state = position({17: 1, 23: 14, 22: -15})
        high = Move(WHITE, 17, 20, 3)
        self.assertEqual([t.moves for t in legal_turns(state, (2, 3))], [(high,)])
        self.assertEqual(turn_options(state, (2, 3))[1], (high,))
        with self.assertRaises(ValueError):
            apply_turn(state, (2, 3), (Move(WHITE, 17, 19, 2),))

    def test_lower_die_allowed_if_higher_die_is_unplayable(self):
        state = position({17: 1, 23: 14, 20: -5, 22: -10})
        turns = legal_turns(state, (2, 3))
        self.assertTrue(turns)
        self.assertTrue(all(tuple(m.die for m in t.moves) == (2,) for t in turns))
        self.assertIn((Move(WHITE, 17, 19, 2),), {t.moves for t in turns})

    def test_doubles_supply_four_individual_moves(self):
        state = position({0: 15}, (0, 15))
        expected = tuple(Move(WHITE, i, i + 3, 3) for i in (0, 3, 6, 9))
        turns = legal_turns(state, (3, 3))
        self.assertTrue(all(len(t.moves) == 4 for t in turns))
        self.assertEqual(apply_turn(state, (3, 3), expected),
                         position({0: 14, 12: 1}, (0, 15), BROWN))
        with self.assertRaises(ValueError):
            apply_turn(state, (3, 3), expected[:2])

    def test_doubles_must_use_all_three_playable_steps(self):
        state = position({14: 1, 23: 14, 22: -15})
        expected = tuple(Move(WHITE, i, i + 2, 2) for i in (14, 16, 18))
        self.assertEqual([t.moves for t in legal_turns(state, (2, 2))], [expected])


class BearingOffTests(unittest.TestCase):
    def test_exact_wins_immediately_with_another_checker_twelve_pips_out(self):
        state = position({12: 14, 22: 1}, (0, 15))
        winning = (Move(WHITE, 22, OFF, 2),)
        self.assertIn(winning, {t.moves for t in legal_turns(state, (2, 4))})
        result = apply_turn(state, (2, 4), winning)
        self.assertEqual(result, position({12: 14}, (0, 15), BROWN))
        self.assertEqual(winner(result), WHITE)
        self.assertEqual(result.white_off, 1)
        self.assertEqual(legal_turns(result, (6, 6)), ())
        with self.assertRaises(ValueError):
            apply_turn(state, (2, 4), winning + (Move(WHITE, 12, 16, 4),))

    def test_crossing_into_home_half_enables_exact_bearoff_on_second_die(self):
        state = position({11: 1, 22: 14}, (0, 15))
        enter_half = Move(WHITE, 11, 12, 1)
        bear_off = Move(WHITE, 22, OFF, 2)
        self.assertEqual(winner(apply_turn(state, (1, 2), (enter_half, bear_off))), WHITE)
        with self.assertRaises(ValueError):
            apply_turn(state, (1, 2), (bear_off,))

    def test_widening_home_half_preserves_pip_distance(self):
        state = position({12: 15}, (0, 15))
        steps = (Move(WHITE, 12, 18, 6), Move(WHITE, 18, OFF, 6))
        self.assertEqual(winner(apply_turn(state, (6, 6), steps)), WHITE)
        with self.assertRaises(ValueError):
            apply_turn(state, (6, 6), (Move(WHITE, 12, OFF, 6),))

    def test_overshoot_only_from_farthest_point(self):
        state = position({22: 7, 23: 8}, (0, 15))
        turns = legal_turns(state, (4, 4))
        self.assertEqual([t.moves for t in turns], [(Move(WHITE, 22, OFF, 4),)])
        with self.assertRaises(ValueError):
            apply_turn(state, (4, 4), (Move(WHITE, 23, OFF, 4),))

    def test_farthest_test_includes_whole_half(self):
        state = position({12: 1, 22: 14}, (0, 15))
        turns = legal_turns(state, (3, 3))
        self.assertTrue(turns)
        self.assertTrue(all(t.moves[0] == Move(WHITE, 12, 15, 3) for t in turns))
        with self.assertRaises(ValueError):
            apply_turn(state, (3, 3), (Move(WHITE, 22, OFF, 3),))

    def test_brown_home_half_bearoff_and_winner(self):
        state = position({2: -1, 11: -14}, (15, 0), BROWN)
        moves = (Move(BROWN, 2, OFF, 3),)
        self.assertEqual(winner(apply_turn(state, (3, 6), moves)), BROWN)
        outside = position({2: -1, 12: -14}, (15, 0), BROWN)
        with self.assertRaises(ValueError):
            apply_turn(outside, (3, 6), moves)

    def test_interactive_bearoff_offers_two_dice_for_same_target(self):
        state = position({23: 15}, (0, 15))
        short, long = Move(WHITE, 23, OFF, 1), Move(WHITE, 23, OFF, 4)
        self.assertEqual(set(turn_options(state, (1, 4))[1]), {short, long})
        for step in (short, long):
            preview, options, done = turn_options(state, (1, 4), (step,))
            self.assertTrue(done)
            self.assertEqual(options, ())
            self.assertEqual(winner(preview), WHITE)
            self.assertEqual(preview.player, WHITE)


class TurnInterfaceTests(unittest.TestCase):
    def test_rollout_samples_each_distinct_result_once(self):
        state = starting_state()
        dice = (1, 2)
        turns = legal_turns(state, dice)
        rng = Mock()
        rng.choice.side_effect = lambda keys: keys[-1]
        self.assertEqual(sample_turn(state, dice, rng), turns[-1])
        offered, = rng.choice.call_args.args
        self.assertEqual(list(offered), [state_key(turn.state) for turn in turns])
        blocked = position({0: -7, 1: -8, 22: 14}, (1, 0))
        self.assertEqual(sample_turn(blocked, dice).moves, ())
        terminal = position({23: 14, 0: -15}, player=BROWN)
        with self.assertRaises(ValueError):
            sample_turn(terminal, dice)

    def test_distinct_opening_positions(self):
        for dice, count in (((1, 2), 48), ((1, 1), 290), ((6, 6), 40)):
            with self.subTest(dice=dice):
                turns = legal_turns(starting_state(), dice)
                self.assertEqual(len(turns), count)
                self.assertEqual(len({t.state for t in turns}), count)

    def test_commuting_die_orders_share_one_outcome_and_preview_keeps_player(self):
        state = position({4: 15}, (0, 15))
        expected = position({4: 14, 7: 1}, (0, 15), BROWN)
        orders = (
            (Move(WHITE, 4, 5, 1), Move(WHITE, 5, 7, 2)),
            (Move(WHITE, 4, 6, 2), Move(WHITE, 6, 7, 1)),
        )
        turns = [t for t in legal_turns(state, (1, 2)) if t.state == expected]
        self.assertEqual(len(turns), 1)
        self.assertIn(turns[0].moves, orders)
        _, options, done = turn_options(state, (1, 2))
        self.assertFalse(done)
        self.assertEqual(set(options), {path[0] for path in orders})
        for path in orders:
            preview, remaining, done = turn_options(state, (1, 2), path[:1])
            self.assertFalse(done)
            self.assertEqual(preview.player, WHITE)
            self.assertIn(path[1], remaining)
            self.assertEqual(turn_options(state, (1, 2), path), (with_next_player(expected), (), True))
            self.assertEqual(apply_turn(state, (1, 2), path), expected)
        self.assertEqual(state, position({4: 15}, (0, 15)))

    def test_no_pass_when_mobile_or_move_using_an_unrolled_die(self):
        for moves in ((), (Move(WHITE, 0, 3, 3), Move(WHITE, 0, 1, 1))):
            with self.subTest(moves=moves), self.assertRaises(ValueError):
                apply_turn(starting_state(), (1, 2), moves)

    def test_color_symmetry_roll_order_and_checker_conservation(self):
        state = position({5: 3, 9: 11, 1: -1, 7: -2, 14: -12}, (1, 0))
        for dice in ((2, 4), (1, 1)):
            with self.subTest(dice=dice):
                turns = legal_turns(state, dice)
                finals = {t.state for t in turns}
                self.assertEqual(len(finals), len(turns))
                self.assertEqual(finals, {t.state for t in legal_turns(state, dice[::-1])})
                self.assertEqual({mirrored(s) for s in finals},
                                 {t.state for t in legal_turns(mirrored(state), dice)})
                for result in finals:
                    self.assertEqual(result.player, BROWN)
                    self.assertEqual(tuple(map(sum, zip(*to_legacy(result)))), (15, 15))

    def test_invalid_roll_rejected(self):
        for dice in ((0, 5), (1, 7), (2,), (2, 2, 2, 2)):
            with self.subTest(dice=dice), self.assertRaises(ValueError):
                legal_turns(starting_state(), dice)


class OriginalEngineParityTests(unittest.TestCase):
    def test_full_outcome_and_first_move_sets_from_original_engine(self):
        """The fixed fixture came from the original 27-cell engine, independently of State."""
        cases = json.loads(Path(__file__).with_name("rule_regressions.json").read_text())
        for case in cases:
            with self.subTest(position=case["name"]):
                state = State(case["points"], *case["bar"], case["player"])
                dice = case["dice"]
                turns = legal_turns(state, dice)
                self.assertEqual(len(turns), case["count"])
                self.assertEqual(len({t.state for t in turns}), len(turns))
                self.assertTrue(all(t.state.player == -state.player for t in turns))
                self.assertEqual(sum(winner(t.state) is not None for t in turns), case["winning"])
                # The fingerprint includes every checker at every index in every final board.
                keys = sorted(bytes(n for cell in to_legacy(t.state) for n in cell) for t in turns)
                self.assertEqual(hashlib.sha256(b"".join(keys)).hexdigest(), case["sha256"])
                if winner(state) is None:
                    _, moves, _ = turn_options(state, dice)
                    first = {(int(m.checker), m.source, m.target, m.die) for m in moves}
                    self.assertEqual(first, set(map(tuple, case["first"])))


if __name__ == "__main__":
    unittest.main()
