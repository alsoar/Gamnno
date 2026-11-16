"""Policy decisions on full-count positions; all candidate paths use the rules engine."""

import unittest

from backgammon import (
    BAR, OFF, BROWN, WHITE, Move, State, Turn, apply_turn, legal_turns,
    starting_state, state_key, turn_options, winner,
)
from heuristic import choose_turn, evaluate_turn, tier


def position(points, brown_bar=0, white_bar=0):
    board = [points.get(i, 0) for i in range(24)]
    return State(board, white_bar, brown_bar, WHITE)


def candidate(state, dice, *moves):
    return Turn(moves, apply_turn(state, dice, moves))


def mirror(state):
    return State(-state.board[::-1], state.brown_bar, state.white_bar, -state.player)


class HeuristicTests(unittest.TestCase):
    def pick(self, state, dice):
        before = state_key(state)
        chosen = choose_turn(state, dice)
        self.assertEqual(apply_turn(state, dice, chosen.moves), chosen.state)
        self.assertEqual(state_key(state), before)
        return chosen, evaluate_turn(state, chosen)

    def test_capture_single_exposed_checker(self):
        state = position({1: 1, 5: 14, 4: -1}, brown_bar=14)
        turn, m = self.pick(state, (1, 3))
        self.assertEqual(turn.moves, (Move(WHITE, 1, 4, 3), Move(WHITE, 4, 5, 1)))
        self.assertEqual((m.captures, m.own_checkers_hit, turn.state.brown_bar), (1, 0, 15))

    def test_prefer_outside_home_quadrant_including_points_in_home_half(self):
        # White can hit on either side of the six-point boundary, 17 or 18.
        state = position({4: 13, 10: 1, 15: 1, 17: -1, 18: -1}, brown_bar=13)
        dice = (2, 3)
        inside = candidate(state, dice, Move(WHITE, 15, 18, 3), Move(WHITE, 10, 12, 2))
        inside_metrics = evaluate_turn(state, inside)
        self.assertEqual((tier(inside_metrics), inside_metrics.inside_home_captures), (2, 1))
        for side in (state, mirror(state)):
            with self.subTest(player=side.player):
                _, m = self.pick(side, dice)
                self.assertEqual((tier(m), m.outside_home_captures, m.inside_home_captures), (1, 1, 0))
                self.assertEqual(m.capture_value, 6)

    def test_accept_free_capture_inside_home_when_no_free_outside_capture_exists(self):
        state = position({16: 1, 14: 14, 19: -1}, brown_bar=14)
        _, m = self.pick(state, (1, 3))
        self.assertEqual((tier(m), m.inside_home_captures, m.exposure_change), (2, 1, 0))

    def test_free_capture_beats_more_valuable_capture_that_opens_a_blot(self):
        state = position({0: 14, 8: 1, 2: -1, 10: -1}, brown_bar=13)
        dice = (2, 3)
        risky = candidate(state, dice, Move(WHITE, 0, 2, 2), Move(WHITE, 2, 5, 3))
        expensive = evaluate_turn(state, risky)
        self.assertEqual((expensive.captures, expensive.capture_value, expensive.exposure_change), (1, 21, 1))
        _, safe = self.pick(state, dice)
        self.assertEqual((tier(safe), safe.captures, safe.capture_value, safe.exposure_change), (1, 1, 13, 0))

    def test_capture_requires_both_dice(self):
        state = position({3: 1, 12: 14, 8: -1}, brown_bar=14)
        self.assertFalse(any(m.checker == WHITE and m.target == 8 for m in turn_options(state, (2, 3))[1]))
        turn, m = self.pick(state, (2, 3))
        self.assertEqual((turn.moves[0].source, turn.moves[1].target), (3, 8))
        self.assertEqual((m.captures, m.forward_progress, m.backward_moves), (1, 5, 0))

    def test_split_opponent_stack_then_capture_remaining_blot(self):
        state = position({3: 1, 10: 14, 6: -2}, brown_bar=13)
        turn, m = self.pick(state, (2, 3))
        self.assertEqual(turn.moves, (Move(BROWN, 6, 8, 2), Move(WHITE, 3, 6, 3)))
        self.assertEqual((m.captures, m.backward_moves, m.forward_progress), (1, 1, 3))

    def test_count_all_capture_events_and_sum_their_value(self):
        state = position({0: 1, 12: 14, 2: -1, 5: -1}, brown_bar=13)
        _, m = self.pick(state, (2, 3))
        self.assertEqual((m.captures, m.outside_home_captures, m.capture_value), (2, 2, 39))

    def test_capture_value_uses_the_position_at_impact_for_both_players(self):
        state = position({7: 1, 14: 14, 6: -1}, brown_bar=14)
        for side in (state, mirror(state)):
            with self.subTest(player=side.player):
                _, m = self.pick(side, (2, 3))
                # On White's turn the Brown blot is pushed from 6 to 9, then hit.
                self.assertEqual((m.captures, m.capture_value, m.forward_progress), (1, 14, 2))

    def test_final_exposure_even_when_capture_temporarily_opens_blot(self):
        state = position({0: 3, 5: 12, 2: -1}, brown_bar=14)
        turn, m = self.pick(state, (2, 3))
        mid, _, _ = turn_options(state, (2, 3), turn.moves[:1])
        self.assertEqual(int((mid.board == WHITE).sum()), 1)
        self.assertEqual((m.exposed_before, m.exposed_after, m.exposure_change, tier(m)), (0, 0, 0, 1))

    def test_reduce_exposure_by_joining_two_blots(self):
        state = position({3: 1, 8: 1, 17: 13}, brown_bar=15)
        turn, m = self.pick(state, (2, 3))
        self.assertEqual(int(turn.state.board[8]), 2)
        self.assertEqual((tier(m), m.captures, m.exposed_before, m.exposed_after), (3, 0, 2, 0))

    def test_maintain_exposure_and_favor_own_forward_movement(self):
        state = position({2: 1, 10: 1, 21: 13, 18: -15})
        _, m = self.pick(state, (2, 3))
        self.assertEqual((tier(m), m.exposed_before, m.exposed_after), (4, 2, 2))
        self.assertEqual((m.forward_progress, m.backward_moves), (5, 0))

    def test_farthest_eligible_checker_precedes_total_progress_in_quiet_fallback(self):
        state = position({8: 9, 10: 2, 18: 1, 23: 3, 13: -10, 15: -2, 20: -3})
        dice = (2, 6)
        faster = candidate(state, dice, Move(WHITE, 10, 16, 6), Move(WHITE, 16, 18, 2))
        alternative = evaluate_turn(state, faster)
        self.assertEqual((tier(alternative), alternative.backward_distances, alternative.forward_progress),
                         (4, (14, 8), 8))
        _, selected = self.pick(state, dice)
        self.assertEqual((tier(selected), selected.backward_distances, selected.forward_progress),
                         (4, (16,), 2))

    def test_equivalent_paths_have_different_backward_distances_and_agent_uses_the_better_one(self):
        state = position({2: 1, 10: 1, 21: 13}, brown_bar=15)
        dice = (2, 3)
        near_first = candidate(state, dice, Move(WHITE, 2, 5, 3), Move(WHITE, 5, 7, 2))
        far_first = candidate(state, dice, Move(WHITE, 2, 4, 2), Move(WHITE, 4, 7, 3))
        self.assertEqual(near_first.state, far_first.state)
        self.assertEqual(sum(t.state == near_first.state for t in legal_turns(state, dice)), 1)
        self.assertEqual(evaluate_turn(state, near_first).backward_distances, (22, 19))
        chosen, m = self.pick(state, dice)
        self.assertEqual(m.backward_distances, (22, 20))
        self.assertEqual(chosen, far_first)

    def test_takeover_preserves_a_played_prefix_and_scores_the_whole_path(self):
        state = position({3: 1, 10: 14, 6: -2}, brown_bar=13)
        split = Move(BROWN, 6, 8, 2)
        best = choose_turn(state, (2, 3), prefix=(split,))
        self.assertEqual(best.moves, (split, Move(WHITE, 3, 6, 3)))
        m = evaluate_turn(state, best)
        self.assertEqual((m.captures, m.backward_moves), (1, 1))
        self.assertEqual(choose_turn(state, (2, 3), prefix=best.moves), best)

    def test_takeover_rejects_a_prefix_that_prevents_compulsory_dice_use(self):
        state = position({5: 1, 8: 14, 1: -2, 4: -1}, brown_bar=12)
        with self.assertRaises(ValueError):
            choose_turn(state, (1, 2), prefix=(Move(BROWN, 4, 5, 1),))

    def test_equivalent_winning_paths_hit_same_checker_at_different_locations(self):
        state = position({12: 14, 18: 1, 20: -1}, brown_bar=14)
        dice = (2, 2)
        race = (Move(WHITE, 18, 20, 2), Move(WHITE, 20, 22, 2), Move(WHITE, 22, OFF, 2))
        earlier = candidate(state, dice, *race)
        later = candidate(state, dice, Move(BROWN, 20, 22, 2), *race)
        self.assertEqual(earlier.state, later.state)
        a, b = evaluate_turn(state, earlier), evaluate_turn(state, later)
        self.assertEqual((a.captures, b.captures, a.capture_value, b.capture_value), (1, 1, 3, 1))

    def test_minimize_extra_exposure_if_every_legal_turn_creates_blots(self):
        state = position({4: 15}, brown_bar=15)
        dice = (1, 2)
        split = candidate(state, dice, Move(WHITE, 4, 5, 1), Move(WHITE, 4, 6, 2))
        self.assertEqual(evaluate_turn(state, split).exposure_change, 2)
        _, m = self.pick(state, dice)
        self.assertEqual((tier(m), m.exposure_change, m.backward_distances), (5, 1, (20, 19)))

    def test_avoid_voluntary_self_hit_when_higher_priority_criteria_are_equal(self):
        state = position({1: 3, 3: 11, 5: 1, 0: -1, 18: -1}, brown_bar=13)
        dice = (5, 4)
        # Self-hit and re-enter. The own bar ends empty, yet the hit event counts.
        self_hit = candidate(state, dice, Move(BROWN, 0, 5, 5), Move(WHITE, BAR, 3, 4))
        sacrifice = evaluate_turn(state, self_hit)
        _, safe = self.pick(state, dice)
        for m in (sacrifice, safe):
            self.assertEqual((tier(m), m.exposed_after, m.captures, m.forward_progress), (3, 0, 0, 4))
        self.assertEqual((sacrifice.own_checkers_hit, safe.own_checkers_hit), (1, 0))

    def test_self_hit_cannot_buy_an_exposure_reduction_without_a_quadrant_capture(self):
        state = position({4: 14, 8: 1, 6: -1}, brown_bar=14)
        dice = (2, 5)
        sacrifice = candidate(state, dice, Move(BROWN, 6, 8, 2), Move(WHITE, BAR, 4, 5))
        m = evaluate_turn(state, sacrifice)
        self.assertEqual((m.own_checkers_hit, m.exposure_change, m.opponent_home_captures), (1, -1, 0))
        _, safe = self.pick(state, dice)
        self.assertEqual((safe.own_checkers_hit, safe.exposure_change), (0, 0))

    def test_self_hit_and_reentry_enable_an_extra_opponent_home_capture_for_each_color(self):
        state = position({8: 1, 12: 14, 6: -1, 2: -1}, brown_bar=13)
        for side, color, source, blot, entry in (
            (state, WHITE, 6, 8, 2), (mirror(state), BROWN, 17, 15, 21)
        ):
            with self.subTest(color=color):
                picked, metrics = self.pick(side, (2, 3))
                self.assertEqual(picked.moves, (Move(-color, source, blot, 2), Move(color, BAR, entry, 3)))
                self.assertEqual((metrics.own_checkers_hit, metrics.opponent_home_captures), (1, 1))

    def test_safe_quadrant_capture_disqualifies_equal_yield_more_valuable_self_hit(self):
        state = position({3: 1, 8: 1, 12: 13, 2: -1, 5: -1, 6: -1}, brown_bar=12)
        dice = (2, 3)
        sacrifice = candidate(state, dice, Move(BROWN, 6, 8, 2), Move(WHITE, BAR, 2, 3))
        m = evaluate_turn(state, sacrifice)
        self.assertEqual((m.own_checkers_hit, m.opponent_home_captures, m.capture_value), (1, 1, 21))
        _, safe = self.pick(state, dice)
        self.assertEqual((safe.own_checkers_hit, safe.opponent_home_captures, safe.capture_value), (0, 1, 18))

    def test_capture_before_a_self_hit_cannot_pay_for_it(self):
        state = position({15: 13, 23: 1, 5: -1, 10: -2, 18: -1, 20: -11}, white_bar=1)
        dice = (5, 6)
        # Entry on 5 hits. To spend the remaining die Brown must then hit White on 23.
        enter = Move(WHITE, BAR, 5, 6)
        later_hit = Move(BROWN, 18, 23, 5)
        self.assertEqual(turn_options(state, dice, (enter,))[1], (later_hit,))
        tempting = evaluate_turn(state, candidate(state, dice, enter, later_hit))
        self.assertEqual((tempting.opponent_home_captures, tempting.own_checkers_hit), (1, 1))
        chosen, safe = self.pick(state, dice)
        self.assertEqual(chosen.moves[0], Move(WHITE, BAR, 4, 5))
        self.assertEqual((safe.opponent_home_captures, safe.own_checkers_hit), (0, 0))

    def test_a_capture_in_opponents_half_but_outside_their_quadrant_is_insufficient(self):
        state = position({9: 1, 16: 14, 6: -2, 8: -1}, brown_bar=12)
        dice = (3, 3)
        sacrifice = candidate(state, dice, Move(BROWN, 6, 9, 3), Move(WHITE, BAR, 2, 3),
                              Move(WHITE, 2, 5, 3), Move(WHITE, 5, 8, 3))
        m = evaluate_turn(state, sacrifice)
        self.assertEqual((m.captures, m.outside_home_captures, m.opponent_home_captures), (1, 1, 0))
        _, safe = self.pick(state, dice)
        self.assertEqual(safe.own_checkers_hit, 0)

    def test_compulsory_self_hit_is_played_if_it_is_the_only_complete_turn(self):
        state = position({9: 2, 15: 9, 19: 2, 23: 1,
                          4: -2, 5: -1, 10: -2, 14: -2, 18: -1, 20: -7}, white_bar=1)
        dice = (5, 6)
        forced = candidate(state, dice, Move(WHITE, BAR, 5, 6), Move(BROWN, 18, 23, 5))
        self.assertEqual(legal_turns(state, dice), (forced,))
        chosen, m = self.pick(state, dice)
        self.assertEqual(chosen, forced)
        self.assertEqual(m.own_checkers_hit, 1)

    def test_one_quadrant_capture_cannot_justify_two_self_hits_even_with_a_prefix(self):
        state = position({9: 1, 11: 1, 16: 13, 2: -1, 6: -3, 8: -3}, brown_bar=8)
        dice = (3, 3)
        prefix = (Move(BROWN, 6, 9, 3), Move(WHITE, BAR, 2, 3))
        double_sacrifice = candidate(state, dice, *prefix, Move(BROWN, 8, 11, 3), Move(WHITE, BAR, 2, 3))
        m = evaluate_turn(state, double_sacrifice)
        self.assertEqual((m.own_checkers_hit, m.opponent_home_captures, m.exposure_change), (2, 1, -2))
        # This remains true after the human commits the first self-hit and its capture.
        for committed in ((), prefix):
            with self.subTest(committed=committed):
                chosen = choose_turn(state, dice, prefix=committed)
                self.assertEqual(chosen.moves[:len(committed)], committed)
                self.assertEqual(apply_turn(state, dice, chosen.moves), chosen.state)
                best = evaluate_turn(state, chosen)
                self.assertEqual((best.own_checkers_hit, best.opponent_home_captures), (1, 1))

    def test_immediate_win_precedes_free_capture(self):
        state = position({12: 13, 14: 1, 22: 1, 16: -1}, brown_bar=14)
        dice = (2, 4)
        capture = candidate(state, dice, Move(WHITE, 14, 16, 2), Move(WHITE, 16, 20, 4))
        self.assertEqual(tier(evaluate_turn(state, capture)), 1)
        turn, m = self.pick(state, dice)
        self.assertTrue(m.wins)
        self.assertEqual((tier(m), winner(turn.state)), (0, WHITE))

    def test_identical_states_and_reversed_roll_produce_the_same_exact_moves(self):
        state = starting_state()
        duplicate = State(state.board.copy(), 0, 0, WHITE)
        first, _ = self.pick(state, (3, 5))
        for other, roll in ((state, (3, 5)), (duplicate, (3, 5)), (duplicate, (5, 3))):
            self.assertEqual(choose_turn(other, roll), first)
        def opposite(point):
            return 23 - point if point < 24 else point
        reflected = tuple(Move(-m.checker, opposite(m.source), opposite(m.target), m.die)
                          for m in first.moves)
        self.assertEqual(choose_turn(mirror(state), (3, 5)).moves, reflected)

    def test_bar_reentry_pass_and_zero_exposure_on_bar(self):
        state = position({7: 14, 0: -2, 2: -2}, brown_bar=11, white_bar=1)
        turn, m = self.pick(state, (1, 3))
        self.assertEqual(turn.moves, ())
        self.assertEqual((m.exposed_before, m.exposed_after, m.forward_progress), (0, 0, 0))
        reenter, m = self.pick(state, (2, 5))
        self.assertEqual(reenter.moves[0].source, BAR)
        self.assertEqual(m.backward_distances[0], 25)

    def test_overshoot_credits_only_remaining_forward_distance(self):
        state = position({23: 15}, brown_bar=15)
        turn = candidate(state, (4, 6), Move(WHITE, 23, OFF, 6))
        m = evaluate_turn(state, turn)
        self.assertEqual((m.wins, m.forward_progress, m.backward_distances), (True, 1, (1,)))
        with self.assertRaises(ValueError):
            choose_turn(turn.state, (2, 5))


if __name__ == "__main__":
    unittest.main()
