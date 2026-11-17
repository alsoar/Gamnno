"""Allocation, replacement, stream separation and complete-turn delegation tests."""
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
import json
from random import Random
import unittest
from unittest.mock import patch

import backgammon as game
from agents import AGENT_NAMES, create_agent
from synthetic_plan import (
    CATEGORIES, CROSS_BUCKETS, DEFAULT_WEIGHTS, PURE, GameMixtureAgent, NoisyPolicy,
    PlanConfig, PolicySpec, TurnMixtureAgent, build_policies, child_seed, make_plan, spec_for,
)


class AllocationTests(unittest.TestCase):
    def test_full_and_pilot_count_completed_slots_and_exact_global_splits(self):
        for n, h, by_category, split_counts in [
            (5000, 100, (2188, 625, 1250, 625, 312), (4000, 500, 500)),
            (80000, 1000, (35000, 10000, 20000, 10000, 5000), (64000, 8000, 8000)),
        ]:
            slots = make_plan(PlanConfig(target_games=n, heldout_games=h))
            self.assertEqual(len({s.slot_id for s in slots}), n + h)
            self.assertEqual(Counter(s.category for s in slots), dict(zip((*CATEGORIES, "heldout"),
                                                                         (*by_category, h))))
            self.assertEqual(Counter(s.split for s in slots), dict(zip(
                ("train", "validation", "test", "heldout_evaluation"), (*split_counts, h))))
            if n == 80000:
                self.assertEqual(Counter(s.subcategory for s in slots if s.category == "cross_strategy"),
                                 dict(zip(CROSS_BUCKETS, (5250, 5250, 5250, 5250, 7000, 5250, 1750))))
        self.assertEqual(make_plan(PlanConfig(target_games=0, heldout_games=0)), [])
        self.assertEqual(make_plan(PlanConfig(target_games=1, heldout_games=0))[0].split, "train")

    def test_cross_groups_disjoint_exhaustive_color_and_opponent_balanced(self):
        cfg = PlanConfig(target_games=1600, heldout_games=0)
        pairs_by_group = defaultdict(Counter)
        directions = defaultdict(Counter)
        for slot in make_plan(cfg):
            if slot.category == "cross_strategy":
                spec = spec_for(slot, 0, cfg)
                pair = tuple(sorted((spec.white_agent, spec.brown_agent)))
                pairs_by_group[spec.subcategory][pair] += 1
                directions[pair][spec.white_agent] += 1
        seen = set()
        for sub in CROSS_BUCKETS:
            group = pairs_by_group[sub]
            self.assertFalse(seen.intersection(group))
            seen.update(group)
            self.assertLessEqual(max(group.values()) - min(group.values()), 1)
        self.assertEqual(seen, {tuple(sorted(pair)) for pair in combinations(AGENT_NAMES, 2)})
        for pair, colors in directions.items():
            self.assertLessEqual(abs(colors[pair[0]] - colors[pair[1]]), 1)
        self.assertEqual(len(pairs_by_group["remaining_nonrandom"]), 14)

    def test_game_types_do_not_drift_and_initial_selfplay_balance(self):
        cfg = PlanConfig(target_games=5000, heldout_games=100)
        kinds = defaultdict(Counter)
        selfplay = Counter()
        rates = Counter()
        for slot in make_plan(cfg):
            s = spec_for(slot, 0, cfg)
            kinds[s.category][s.subcategory] += 1
            if s.white_agent == s.brown_agent == "random":
                self.assertEqual(s.category, "exploratory")
            if s.category == "pure_selfplay":
                self.assertEqual(s.white_agent, s.brown_agent)
                self.assertIn(s.white_agent, PURE)
                selfplay[s.white_agent] += 1
            if s.subcategory == "noisy_vs_pure":
                seat = next(desc for desc in (s.white, s.brown) if desc.kind == "noisy")
                rates[seat.epsilon] += 1
            if s.split == "heldout_evaluation":
                self.assertEqual(s.category, "heldout")
                if s.subcategory == "heldout_noisy_vs_noisy":
                    self.assertEqual(s.white.kind, s.brown.kind, "noisy")
                    self.assertNotEqual(s.white.base, s.brown.base)
                else:
                    self.assertTrue(s.white.heldout and s.brown.heldout)
        self.assertEqual(set(selfplay), set(PURE))
        self.assertLessEqual(max(selfplay.values()) - min(selfplay.values()), 1)
        self.assertEqual(set(rates), set(cfg.epsilons))
        self.assertLessEqual(max(rates.values()) - min(rates.values()), 2)
        for category, expected in [
            ("turn_mixture", {"turn_vs_turn", "turn_vs_pure"}),
            ("game_mixture", {"game_vs_game", "game_vs_turn", "game_vs_pure"}),
            ("exploratory", {"noisy_vs_pure", "random_vs_random", "random_vs_pure"}),
            ("heldout", {"heldout_turn_vs_game", "heldout_noisy_vs_noisy"}),
        ]:
            self.assertEqual(set(kinds[category]), expected)
            self.assertLessEqual(max(kinds[category].values()) - min(kinds[category].values()), 1)

    def test_replacements_rotate_in_their_original_stratum_with_fresh_seeds(self):
        cfg = PlanConfig(target_games=400, heldout_games=5, max_attempts_per_slot=32)
        slots = make_plan(cfg)
        for category, sub, matchups in [
            ("pure_selfplay", "all", 8), ("cross_strategy", "optimizer_nonrandom", 7),
            ("cross_strategy", "sprinter_hoarder_fortress", 2),
        ]:
            slot = next(s for s in slots if (s.category, s.subcategory) == (category, sub))
            attempts = [spec_for(slot, i, cfg) for i in range(2 * matchups)]
            self.assertEqual(len({s.game_id for s in attempts}), len(attempts))
            self.assertEqual(len({s.dice_seed for s in attempts}), len(attempts))
            pairs = {tuple(sorted((s.white_agent, s.brown_agent))) for s in attempts[:matchups]}
            self.assertEqual(len(pairs), matchups)
            self.assertEqual({(s.split, s.category) for s in attempts}, {(slot.split, category)})
            for first, second in zip(attempts, attempts[matchups:]):
                self.assertEqual((first.white, first.brown), (second.brown, second.white))
        with self.assertRaises(ValueError):
            spec_for(slots[0], -1, cfg)
        with self.assertRaises(ValueError):
            spec_for(slots[0], cfg.max_attempts_per_slot, cfg)

    def test_schedule_independent_of_dispatch_order_resume_budget_and_mapping_roundtrip(self):
        cfg = PlanConfig(target_games=140, heldout_games=9, mixture_weights={"sprinter": 2, "baseline": 1})
        slots = make_plan(cfg)
        encoded = json.loads(json.dumps(cfg.as_dict()))
        resumed = replace(cfg, max_attempts_per_slot=100)
        self.assertEqual(cfg.identity, resumed.identity)
        self.assertEqual(slots, make_plan({"plan": encoded, "workers": 17}))
        self.assertEqual(slots, make_plan(resumed))
        ordered = {s.slot_id: spec_for(s, 2, cfg) for s in slots}
        self.assertEqual(ordered, {s.slot_id: spec_for(s, 2, resumed) for s in reversed(slots)})
        self.assertEqual(ordered, {s.slot_id: spec_for(s, 2, encoded) for s in slots})
        identifiers = {s.game_id for s in ordered.values()}
        self.assertEqual(len(identifiers), len(slots))
        alternate = replace(cfg, namespace="full")
        ids2 = {spec_for(s, 2, alternate).game_id for s in make_plan(alternate)}
        self.assertFalse(identifiers.intersection(ids2))
        self.assertNotEqual(cfg.identity, replace(cfg, max_turns=321).identity)
        streams = [value for s in ordered.values() for value in
                   (s.generation_seed, s.dice_seed, s.sampling_seed, s.trajectory_seed)]
        self.assertEqual(len(streams), len(set(streams)))
        self.assertTrue(all(0 <= i < 2**63 for i in streams))

    def test_config_validation(self):
        for overrides in ({"target_games": -1}, {"heldout_games": -1}, {"master_seed": True},
                          {"max_attempts_per_slot": 0}, {"max_turns": 0}, {"namespace": ""},
                          {"epsilons": []}, {"epsilons": [2]}, {"epsilons": [.1, .1]},
                          {"mixture_weights": {"not_an_agent": 1}},
                          {"mixture_weights": [0] * 9}, {"mixture_weights": [-1] * 9},
                          {"mixture_weights": [float("nan")] * 9}, {"mixture_weights": [1]},
                          {"heldout_weights": DEFAULT_WEIGHTS}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                PlanConfig(**overrides)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PlanConfig(target_games=70, heldout_games=3)
        slot = make_plan(self.cfg)[0]
        self.spec = spec_for(slot, 0, self.cfg)
        self.state = game.starting_state()

    def test_fixed_and_noisy_endpoints_delegate_to_existing_complete_turns(self):
        for eps, kind, expected in [(None, "pure", "baseline"), (0., "noisy", "baseline"),
                                    (1., "noisy", "random")]:
            desc = PolicySpec(kind, "baseline", eps)
            seat, _ = build_policies(replace(self.spec, white=desc), self.cfg)
            seed = child_seed(child_seed(self.spec.generation_seed, "white"), "random-legal-turn")
            control = create_agent(expected, seed=0)
            control_rng = Random(seed)
            for dice in [(4, 2), (3, 5), (1, 2)]:
                candidate = seat.choose_candidate(self.state, dice, rng=Random(999))
                self.assertEqual(candidate.turn, control.choose_turn(self.state, dice, rng=control_rng))
                self.assertEqual(seat.selected_strategy, expected)
                self.assertEqual(seat.explored, eps == 1)
            self.assertEqual(seat.metadata["base"], "baseline")
            self.assertEqual(seat.metadata["epsilon"], eps)

    def test_one_selection_per_roll_and_independent_game_draws(self):
        weights = {"sprinter": 1, "baseline": 1}
        cfg = replace(self.cfg, mixture_weights=weights)
        spec = replace(self.spec, white=PolicySpec("turn"), brown=PolicySpec("game"))
        a, b = build_policies(spec, cfg)
        self.assertIsInstance(a, TurnMixtureAgent)
        self.assertIsInstance(b, GameMixtureAgent)
        fixed = b.game_strategy
        self.assertEqual(b.metadata["game_strategy"], fixed)
        self.assertEqual(a.metadata["mixture_weights"], {k: .5 if k in weights else 0. for k in AGENT_NAMES})
        with patch.object(a, "_draw", wraps=a._draw) as turn_draw, patch.object(b, "_draw", wraps=b._draw) as game_draw:
            seen = set()
            for _ in range(8):
                turn = a.choose_turn(self.state, (1, 2))
                seen.add(a.selected_strategy)
                self.assertEqual(turn, create_agent(a.selected_strategy).choose_turn(self.state, (1, 2)))
                b.choose_turn(self.state, (2, 3))
                self.assertEqual(b.selected_strategy, fixed)
            self.assertEqual(turn_draw.call_count, 8)
            self.assertEqual(game_draw.call_count, 0)
        self.assertEqual(seen, set(weights))
        desc = PolicySpec("game")
        pairs = [build_policies(replace(spec, white=desc, brown=desc, generation_seed=i), cfg)
                 for i in range(20)]
        self.assertTrue(any(w.game_strategy != b.game_strategy for w, b in pairs))
        self.assertTrue(all(w.game_strategy in weights and b.game_strategy in weights for w, b in pairs))

    def test_noisy_coin_and_seat_strategy_independent_of_action_rng_and_other_seat(self):
        desc = PolicySpec("noisy", "baseline", .3)
        spec = replace(self.spec, white=desc, brown=PolicySpec("turn"))
        first, other = build_policies(spec, self.cfg)
        mirror, _ = build_policies(spec, self.cfg)
        self.assertIsInstance(first, NoisyPolicy)
        flips = []
        for i in range(18):
            # Consumption of random-move draws cannot change the exploration coin.
            for _ in range(i % 4):
                first._actions.random()
            first.choose_turn(self.state, (2, 3), rng=Random(i))
            if i % 2 == 0:
                other.choose_turn(self.state, (1, 3))
            mirror.choose_turn(self.state, (2, 3), rng=Random(i + 1000))
            self.assertEqual(first.explored, mirror.explored)
            self.assertEqual(first.selected_strategy, mirror.selected_strategy)
            flips.append(first.explored)
        self.assertEqual(set(flips), {True, False})

    def test_heldout_wrappers_use_distinct_probabilities(self):
        cfg = replace(self.cfg, mixture_weights={"sprinter": 1}, heldout_weights={"hoarder": 1})
        for kind in ("game", "turn"):
            s = replace(self.spec, white=PolicySpec(kind, heldout=True), brown=PolicySpec(kind))
            held, primary = build_policies(s, cfg)
            held.choose_turn(self.state, (1, 4))
            primary.choose_turn(self.state, (1, 4))
            self.assertEqual((held.selected_strategy, primary.selected_strategy), ("hoarder", "sprinter"))
            self.assertNotEqual(held.name, primary.name)
            json.dumps(held.metadata, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
