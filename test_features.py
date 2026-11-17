"""Full-turn strategic features on physical, full-count engine positions."""
import json
import unittest
from unittest.mock import patch

import backgammon as game
import features as f


def position(pieces, white_bar=0, brown_bar=0):
    return game.State([pieces.get(i, 0) for i in range(24)], white_bar, brown_bar, game.WHITE)


def explain(state, dice, *moves):
    turn = game.Turn(moves, game.apply_turn(state, dice, moves))
    return f.evaluate_chosen(state, turn)


def mirror(state):
    return game.State(-state.board[::-1], state.brown_bar, state.white_bar, -state.player)


def mirror_move(m):
    return game.Move(-m.checker, 23-m.source if m.source < 24 else m.source,
                     23-m.target if m.target < 24 else m.target, m.die)


class FeatureTests(unittest.TestCase):
    def test_named_position_helpers_use_requested_player_and_opponent(self):
        state = position({3: 2, 4: 2, 5: 2, 16: 8, 18: 1, 9: -15})
        own = f.side_features(state, game.WHITE)
        for name in ("blots", "outside_home_half", "distance_to_home_half", "leading_checker_distance",
                     "occupied_points", "made_points", "stack_concentration", "longest_blockade"):
            with self.subTest(accessor=name):
                self.assertEqual(getattr(f, name)(state, game.WHITE), getattr(own, name))
        self.assertEqual(f.effective_vulnerability(state, game.WHITE, bar_penalty=2), own.blots)
        self.assertEqual(f.opponent_forward_mobility(state, game.WHITE), .5)
        self.assertEqual(f.opponent_forward_mobility(state, game.BROWN), own.forward_mobility)
        self.assertEqual(f.capture_opportunities(state, game.WHITE), own.threats)

    def test_named_turn_helpers_accept_cached_candidate_or_initial_state_and_turn(self):
        state = position({8: 1, 12: 14, 6: -1, 2: -1}, brown_bar=13)
        c = explain(state, (2, 3), game.Move(-1, 6, 8, 2), game.Move(1, game.BAR, 2, 3))
        for name, expected in (("captures", 1), ("self_captures", 1), ("outside_home_captures", 1),
                               ("capture_value", 21), ("total_forward_progress", 3),
                               ("backward_displacement", 2)):
            with self.subTest(accessor=name):
                fn = getattr(f, name)
                self.assertEqual(fn(state, c.turn), expected)
                with patch.object(f, "evaluate_chosen", side_effect=AssertionError("recomputed a candidate")):
                    self.assertEqual(fn(c), expected)
                    self.assertEqual(fn(c, game.WHITE), expected)
        self.assertEqual(f.exposure_change(state, c.turn.state, game.WHITE), c.exposure_change)
        self.assertFalse(f.immediate_win(c, game.WHITE))
        winning = position({22: 15}, brown_bar=15)
        end = explain(winning, (2, 3), game.Move(1, 22, game.OFF, 2))
        self.assertTrue(f.immediate_win(end.turn, game.WHITE))
        self.assertFalse(f.immediate_win(end.turn, game.BROWN))
        self.assertTrue(f.immediate_win(end.turn))
        with self.assertRaises(ValueError):
            f.capture_value(c, game.BROWN)

    def test_bar_and_half_geometry_structure_and_custom_vulnerability(self):
        state = position({2: 2, 12: 3, 13: 7, 20: 1}, white_bar=2, brown_bar=15)
        own = f.side_features(state, game.WHITE, bar_penalty=4)
        self.assertEqual((own.bar, own.blots, own.effective_vulnerability), (2, 1, 9))
        self.assertEqual((own.outside_home_half, own.distance_to_home_half), (4, 46))
        self.assertEqual((own.eligible, own.bearoff_candidates, own.leading_checker_distance), (False, 0, 4))
        self.assertEqual((own.occupied_points, own.made_points, own.stack_concentration), (4, 3, 63))
        self.assertEqual((own.longest_blockade, own.rear_home_stack), (2, 6*9 + 5*49))
        self.assertEqual(own.blockade_pairs, 1 << 12)
        self.assertEqual(own, f.side_features(mirror(state), game.BROWN, bar_penalty=4))
        self.assertEqual(f.side_features(state, game.BROWN).leading_checker_distance, 25)
        for penalty in (-1, float("nan"), float("inf")):
            with self.subTest(penalty=penalty), self.assertRaises(ValueError):
                f.side_features(state, game.WHITE, bar_penalty=penalty)

    def test_eligibility_vs_immediate_bearoff_opportunity(self):
        full = position({12: 13, 20: 2}, brown_bar=15)
        s = f.side_features(full, game.WHITE)
        self.assertEqual((s.eligible, s.bearoff_candidates, s.leading_checker_distance), (True, 2, 4))
        far = f.side_features(position({12: 15}, brown_bar=15), game.WHITE)
        self.assertEqual((far.eligible, far.bearoff_candidates, far.leading_checker_distance), (True, 0, 12))
        win = explain(full, (4, 6), game.Move(1, 20, game.OFF, 4))
        self.assertEqual((win.wins, win.own_after.leading_checker_distance), (True, 0))

    def test_splitting_pair_enables_capture_subsequent_threats_include_setup(self):
        state = position({3: 1, 10: 14, 6: -2}, brown_bar=13)
        moves = (game.Move(-1, 6, 8, 2), game.Move(1, 3, 6, 3))
        c = explain(state, (2, 3), *moves)
        self.assertEqual(c.own_before.threats, 1)
        self.assertEqual((c.captures, c.self_captures, c.capture_value), (1, 0, 17))
        self.assertEqual((c.forward_progress, c.backward_displacement, c.backward_moves), (3, 2, 1))
        self.assertEqual((c.events[0].step, c.events[0].move), (1, moves[1]))
        self.assertEqual(c.own_before.blots, c.own_after.blots)

    def test_value_uses_impact_location_and_disruption_bonuses(self):
        state = position({7: 1, 14: 14, 6: -1}, brown_bar=14)
        c = explain(state, (2, 3), game.Move(-1, 6, 9, 3), game.Move(1, 7, 9, 2))
        self.assertEqual((c.events[0].progress, c.capture_value), (14, 14))
        self.assertEqual(c.own_before.threats, 1)  # The untouched enemy blot starts behind White.
        near = position({0: 1, 15: 14, 2: -1, 8: -14})
        c = explain(near, (1, 2), game.Move(1, 0, 2, 2), game.Move(1, 2, 3, 1))
        hit, = c.events
        self.assertEqual((hit.progress, hit.breaks_eligibility, hit.delays_win), (21, True, True))
        self.assertEqual((c.opponent_home_captures, c.capture_value), (1, 39))
        last = position({11: 1, 18: 14, 10: -14, 13: -1})
        c = explain(last, (1, 2), game.Move(1, 11, 13, 2), game.Move(1, 13, 14, 1))
        self.assertTrue(c.events[0].last_outsider)
        self.assertFalse(c.events[0].breaks_eligibility)
        self.assertEqual(c.capture_value, 16)

    def test_self_hit_then_reentry_counts_both_events(self):
        state = position({8: 1, 12: 14, 6: -1, 2: -1}, brown_bar=13)
        dice = (2, 3)
        moves = (game.Move(-1, 6, 8, 2), game.Move(1, game.BAR, 2, 3))
        c = explain(state, dice, *moves)
        self.assertEqual((c.captures, c.self_captures, c.own_after.bar), (1, 1, 0))
        self.assertEqual([(h.step, h.self_hit, h.value) for h in c.events], [(0, True, 0), (1, False, 21)])
        self.assertEqual((c.backward_distances, c.forward_progress), ((25,), 3))
        r = explain(mirror(state), dice, *map(mirror_move, moves))
        self.assertEqual((r.own_after, r.opp_after, r.capture_value), (c.own_after, c.opp_after, c.capture_value))
        self.assertEqual(c, next(x for x in f.iter_candidates(state, dice, prefix=moves[:1]) if x.turn.moves == moves))

    def test_bar_hit_does_not_improve_security_or_home_half_distance(self):
        state = position({8: 1, 12: 14, 6: -15})
        c = explain(state, (1, 2), game.Move(-1, 6, 7, 1), game.Move(-1, 6, 8, 2))
        self.assertEqual((c.own_before.blots, c.own_after.blots, c.own_after.bar), (1, 0, 1))
        self.assertGreater(c.own_after.effective_vulnerability, c.own_before.effective_vulnerability)
        self.assertGreater(c.own_after.distance_to_home_half, c.own_before.distance_to_home_half)
        self.assertEqual(c.own_after.outside_home_half, c.own_before.outside_home_half)

    def test_mobility_and_nearby_blockade(self):
        state = position({3: 2, 4: 2, 5: 2, 16: 9, 9: -15})
        own = f.side_features(state, game.WHITE)
        self.assertEqual((own.longest_blockade, own.nearby_blocks), (3, 3))
        self.assertEqual(f.side_features(state, game.BROWN).forward_mobility, .5)
        state = position({0: -2, 2: -2, 6: -11, 9: 14}, white_bar=1)
        own = f.side_features(state, game.WHITE)
        self.assertAlmostEqual(own.forward_mobility, 4/6)
        self.assertEqual(own.threats, 0)

    def test_forced_pass_and_terminal(self):
        state = position({0: -2, 2: -2, 6: -11, 9: 14}, white_bar=1)
        c, = f.iter_candidates(state, (1, 3))
        self.assertEqual(c.turn.moves, ())
        self.assertEqual((c.own_before, c.opp_before), (c.own_after, c.opp_after))
        self.assertEqual((c.captures, c.self_captures, c.forward_progress, c.backward_moves), (0, 0, 0, 0))
        self.assertEqual(c, f.evaluate_chosen(state, c.turn))
        self.assertEqual(list(f.iter_candidates(position({12: 14}, brown_bar=15), (2, 2))), [])

    def test_all_doubles_paths_keep_history_while_reusing_endpoint_features(self):
        state = position({2: 15}, brown_bar=15)
        with patch.object(f, "_both", wraps=f._both) as calls:
            rolls = list(f.iter_candidates(state, (2, 2)))
            finals = {x.turn.state for x in rolls}
            self.assertEqual(calls.call_count, len(finals) + 1)
        self.assertGreater(len(rolls), len(finals))
        self.assertTrue(all(len(x.turn.moves) == 4 for x in rolls))
        self.assertEqual(finals, {t.state for t in game.legal_turns(state, (2, 2))})
        with patch.object(f, "_legal_paths", side_effect=AssertionError("regenerated turns")):
            for roll in rolls:
                self.assertEqual(f.evaluate_chosen(state, roll.turn), roll)

    def test_json_record_is_independent_of_engine_objects(self):
        state = position({3: 1, 10: 14, 6: -2}, brown_bar=13)
        c = explain(state, (2, 3), game.Move(-1, 6, 8, 2), game.Move(1, 3, 6, 3))
        record = json.loads(json.dumps(f.feature_record(c), allow_nan=False))
        self.assertNotIn("turn", record)
        self.assertEqual(record["events"][0]["move"]["target"], 6)
        self.assertEqual(record["opp_after"]["bar"], 14)

    def test_prefer_win_returns_first_canonical_win_and_honors_committed_prefix(self):
        state = position({12: 14, 22: 1}, brown_bar=15)
        dice = (2, 3)
        full = list(f.iter_candidates(state, dice))
        expected = next(c for c in full if c.wins)
        with patch.object(f, "_candidate", wraps=f._candidate) as measured:
            self.assertEqual(list(f.iter_candidates(state, dice, prefer_win=True)), [expected])
            self.assertEqual(measured.call_count, 1)
        # Spending die 2 on another checker makes the exact win unavailable.
        prefix = (game.Move(1, 12, 14, 2),)
        allowed = [c for c in full if c.turn.moves[:1] == prefix]
        self.assertTrue(allowed)
        self.assertFalse(any(c.wins for c in allowed))
        self.assertEqual(list(f.iter_candidates(state, dice, prefix=prefix, prefer_win=True)), allowed)


if __name__ == "__main__":
    unittest.main()
