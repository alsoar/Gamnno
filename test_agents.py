"""Strategy choices and sacrifice safeguards on legal complete-turn alternatives."""

from random import Random
import unittest

import backgammon as game
import features
from agents import AGENT_NAMES, AgentConfig, create_agent
from agents.policies import dominates


def position(stacks, white_bar=0, brown_bar=0):
    return game.State([stacks.get(i, 0) for i in range(24)], white_bar, brown_bar, game.WHITE)


def mirrored(state):
    return game.State(-state.board[::-1], state.brown_bar, state.white_bar, -state.player)


def mirror_move(m):
    return game.Move(-m.checker, 23 - m.source if m.source < 24 else m.source,
                     23 - m.target if m.target < 24 else m.target, m.die)


def sacrifice_position():
    # White can push Brown 9->12 onto White's blot and re-enter on Brown's
    # advanced blot at 3. Another capture (worth 10 vs 20) avoids all self-hits.
    return position({10: 14, 12: 1, 9: -8, 3: -1, 14: -6}), (3, 4)


class StrategyTests(unittest.TestCase):
    def test_registered_and_immutable_input(self):
        self.assertEqual(AGENT_NAMES, ("baseline", "sprinter", "fortress", "saboteur", "assassin",
                                       "blockader", "optimizer", "hoarder", "random"))
        state = position({3: 1, 10: 14, 6: -2}, brown_bar=13)
        before = game.state_key(state)
        for name in AGENT_NAMES:
            with self.subTest(agent=name):
                agent = create_agent(name, seed=2026)
                turn = agent.choose_turn(state, (2, 3))
                self.assertEqual(agent.name, name)
                self.assertEqual(game.apply_turn(state, (2, 3), turn.moves), turn.state)
                self.assertEqual(game.state_key(state), before)
                self.assertTrue(agent.explain(state, turn))
        with self.assertRaises(ValueError):
            create_agent("missing")
        with self.assertRaises(ValueError):
            create_agent("baseline", {"bar_penalty": float("nan")})

    def test_baseline_split_then_capture_without_added_vulnerability(self):
        state = position({3: 1, 10: 14, 6: -2}, brown_bar=13)
        c = create_agent("baseline").choose_candidate(state, (2, 3))
        self.assertEqual(c.turn.moves, (game.Move(-1, 6, 8, 2), game.Move(1, 3, 6, 3)))
        self.assertEqual((c.captures, c.outside_home_captures, c.backward_moves), (1, 1, 1))
        self.assertLessEqual(c.own_after.effective_vulnerability, c.own_before.effective_vulnerability)

    def test_sprinter_reaches_eligibility_while_fortress_maintains_a_safe_stack(self):
        state = position({11: 2, 16: 13, 1: -15})
        race = create_agent("sprinter").choose_candidate(state, (1, 2))
        safe = create_agent("fortress").choose_candidate(state, (1, 2))
        self.assertTrue(race.own_after.eligible)
        self.assertFalse(safe.own_after.eligible)
        self.assertEqual(safe.own_after.effective_vulnerability, 0)
        self.assertGreater(race.own_after.effective_vulnerability, 0)

    def test_saboteur_removes_an_eligible_opponents_status(self):
        state = position({10: 15, 11: -2, 8: -11, 3: -2})
        candidates = list(features.iter_candidates(state, (3, 4)))
        selected = create_agent("saboteur").select_candidate(candidates)
        self.assertTrue(selected.opp_before.eligible)
        self.assertFalse(selected.opp_after.eligible)
        self.assertEqual(selected.opp_after.distance_to_home_half, max(c.opp_after.distance_to_home_half for c in candidates))
        self.assertTrue(any(c.opp_after.eligible for c in candidates))

    def test_assassin_takes_a_capture_despite_an_extra_blot(self):
        state = position({10: 15, 11: -2, 8: -11, 3: -2})
        a = create_agent("assassin").choose_candidate(state, (3, 4))
        b = create_agent("baseline").choose_candidate(state, (3, 4))
        self.assertEqual((a.captures, b.captures), (1, 0))
        self.assertGreater(a.own_after.effective_vulnerability, a.own_before.effective_vulnerability)

    def test_blockader_builds_a_run_and_hoarder_joins_a_stack(self):
        state = position({3: 2, 5: 2, 6: 1, 8: 10, 19: -15})
        block = create_agent("blockader").choose_candidate(state, (1, 2))
        self.assertEqual((block.own_before.longest_blockade, block.own_after.longest_blockade), (1, 2))
        state = position({7: 2, 13: 7, 15: 6, 4: -15})
        c = create_agent("hoarder").choose_candidate(state, (2, 4))
        self.assertGreater(c.stack_joins, 0)
        self.assertGreater(c.own_after.stack_concentration, c.own_before.stack_concentration)
        self.assertLess(c.own_after.outside_home_half, c.own_before.outside_home_half)

    def test_optimizer_uses_its_numeric_score(self):
        state = position({7: 2, 13: 7, 15: 6, 4: -15})
        cs = list(features.iter_candidates(state, (2, 4)))
        self.assertTrue(all(not c.wins and not c.self_captures for c in cs))
        a = create_agent("optimizer", AgentConfig(displacement_weight=15))
        picked = a.select_candidate(cs)
        self.assertEqual(a.score(picked), max(a.score(c) for c in cs))
        self.assertGreater(picked.backward_displacement, 0)

    def test_all_deterministic_agents_take_first_canonical_win(self):
        state = position({20: 14, 23: 1, 0: -15})
        cs = list(features.iter_candidates(state, (1, 2)))
        self.assertTrue(any(not c.wins for c in cs))
        first = next(c for c in cs if c.wins).turn
        for name in AGENT_NAMES[:-1]:
            with self.subTest(name=name):
                self.assertEqual(create_agent(name).choose_turn(state, (1, 2)), first)

    def test_symmetry_determinism_and_fixed_prefix(self):
        s, dice = sacrifice_position()
        swapped = mirrored(s)
        for name in AGENT_NAMES[:-1]:
            with self.subTest(name=name):
                a = create_agent(name)
                left = a.choose_turn(s, dice, rng=Random(1))
                self.assertEqual(a.choose_turn(s, tuple(reversed(dice)), rng=Random(20)), left)
                right = a.choose_turn(swapped, dice)
                self.assertEqual(right.moves, tuple(mirror_move(m) for m in left.moves))
                self.assertEqual(right.state, mirrored(left.state))
                prefix = game.turn_options(s, dice)[1][-1:]
                constrained = a.choose_turn(s, dice, prefix=prefix)
                self.assertEqual(constrained.moves[:1], prefix)
                self.assertEqual(game.apply_turn(s, dice, constrained.moves), constrained.state)


class SacrificeTests(unittest.TestCase):
    def test_dominated_self_hit_cannot_remove_a_blot_or_consolidate_a_trailer(self):
        state = position({4: 1, 14: 14, 1: -15})
        cs = list(features.iter_candidates(state, (1, 3)))
        harmful = next(c for c in cs if c.self_captures and c.own_after.bar and not c.own_after.blots)
        self.assertLess(harmful.own_after.occupied_points, harmful.own_before.occupied_points)
        self.assertGreater(harmful.own_after.effective_vulnerability, harmful.own_before.effective_vulnerability)
        self.assertGreaterEqual(harmful.own_after.outside_home_half, harmful.own_before.outside_home_half)
        self.assertGreater(harmful.own_after.distance_to_home_half, harmful.own_before.distance_to_home_half)
        fortress = create_agent("fortress")
        victim = fortress.comparison(harmful, fortress.rank(harmful))
        self.assertTrue(any(dominates(fortress.comparison(c, fortress.rank(c)), victim)
                            for c in cs if not c.self_captures))
        for name in ("fortress", "hoarder", "sprinter", "baseline", "saboteur", "blockader", "assassin", "optimizer"):
            with self.subTest(name=name):
                self.assertFalse(create_agent(name).select_candidate(cs).self_captures)

    def test_assassin_and_baseline_can_choose_a_justified_intermediate_sacrifice(self):
        s, dice = sacrifice_position()
        cs = list(features.iter_candidates(s, dice))
        for name in ("assassin", "baseline"):
            with self.subTest(name=name):
                c = create_agent(name).select_candidate(cs)
                self.assertEqual(c.turn.moves, (game.Move(-1, 9, 12, 3), game.Move(1, game.BAR, 3, 4)))
                self.assertEqual((c.self_captures, c.own_after.bar, c.opponent_home_captures), (1, 0, 1))
                self.assertEqual((c.events[0].self_hit, c.events[1].opponent_home), (True, True))
                self.assertGreater(c.capture_value, max(k.capture_value for k in cs if not k.self_captures))
        self.assertFalse(create_agent("fortress").select_candidate(cs).self_captures)
        self.assertFalse(create_agent("assassin", {"assassin_value_gain": 50}).select_candidate(cs).self_captures)

    def test_baseline_rejects_a_self_hit_after_an_earlier_home_capture(self):
        s = position({15: 13, 23: 1, 5: -1, 10: -2, 18: -1, 20: -11}, white_bar=1)
        cs = list(features.iter_candidates(s, (5, 6)))
        self.assertTrue(any(c.self_captures and c.opponent_home_captures for c in cs))
        self.assertFalse(create_agent("baseline").select_candidate(cs).self_captures)

    def test_incidental_self_hit_in_fixed_prefix_cannot_exempt_a_new_sacrifice(self):
        s = position({9: 1, 11: 1, 16: 13, 2: -1, 6: -3, 8: -3}, brown_bar=8)
        dice = (3, 3)
        prefix = (game.Move(-1, 6, 9, 3), game.Move(1, game.BAR, 2, 3))
        cs = list(features.iter_candidates(s, dice, prefix=prefix))
        self.assertEqual({c.self_captures for c in cs}, {1, 2})
        for name in ("baseline", "fortress", "assassin"):
            with self.subTest(name=name):
                selected = create_agent(name).choose_candidate(s, dice, prefix=prefix)
                self.assertEqual(selected.turn.moves[:2], prefix)
                self.assertEqual((selected.self_captures, selected.opponent_home_captures), (1, 1))
                self.assertEqual(game.apply_turn(s, dice, selected.turn.moves), selected.turn.state)

    def test_optimizer_requires_strict_improvement_after_self_capture_penalty(self):
        s, dice = sacrifice_position()
        cs = list(features.iter_candidates(s, dice))
        weights = dict(capture_weight=24, vulnerability_weight=0, forward_weight=0, displacement_weight=0,
                       blockade_weight=0, victory_weight=0, self_capture_weight=2)
        a = create_agent("optimizer", dict(weights, sacrifice_threshold=7.99))
        costly = a.select_candidate(cs)
        best_clean = max(a.score(c) for c in cs if not c.self_captures)
        self.assertEqual((costly.self_captures, a.score(costly) - best_clean), (1, 8))
        self.assertFalse(create_agent("optimizer", dict(weights, sacrifice_threshold=8)).select_candidate(cs).self_captures)

    def test_every_agent_can_play_the_forced_self_hit(self):
        s = position({9: 2, 15: 9, 19: 2, 23: 1, 4: -2, 5: -1, 10: -2, 14: -2,
                      18: -1, 20: -7}, white_bar=1)
        only, = game.legal_turns(s, (5, 6))
        self.assertEqual(features.evaluate_chosen(s, only).self_captures, 1)
        for name in AGENT_NAMES:
            self.assertEqual(create_agent(name).choose_turn(s, (5, 6)), only)


class RandomTests(unittest.TestCase):
    def test_seeded_samples_reproduce_but_vary(self):
        s, dice = sacrifice_position()
        a, b = create_agent("random", seed=73), create_agent("random", seed=73)
        draws = [a.choose_turn(s, dice) for _ in range(36)]
        self.assertEqual(draws, [b.choose_turn(s, dice) for _ in range(36)])
        self.assertGreater(len({t.state for t in draws}), 1)
        self.assertTrue(all(game.apply_turn(s, dice, t.moves) == t.state for t in draws))

    def test_sampler_offers_every_distinct_state_once_even_when_an_immediate_win_exists(self):
        class Pick:
            def __init__(self, index):
                self.index = index

            def choice(self, items):
                self.n = len(items)
                return items[self.index]

        for state, dice in (sacrifice_position(), (position({20: 14, 23: 1, 0: -15}), (1, 2))):
            legal = game.legal_turns(state, dice)
            expected = {t.state for t in legal}
            seen = set()
            a = create_agent("random")
            for i in range(len(expected)):
                controlled = Pick(i)
                seen.add(a.choose_turn(state, dice, rng=controlled).state)
                self.assertEqual(controlled.n, len(expected))
            self.assertEqual(seen, expected)
            flags = [features.evaluate_chosen(state, t) for t in legal]
            self.assertTrue(any(c.self_captures for c in flags) or
                            (any(c.wins for c in flags) and any(not c.wins for c in flags)))


if __name__ == "__main__":
    unittest.main()
