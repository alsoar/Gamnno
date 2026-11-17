"""Phase, pre-roll, reservoir, and trajectory regressions for the synthetic dataset."""

from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import backgammon as game
from agents import AGENT_NAMES, create_agent
from features import evaluate_chosen
from simulation import play_game
from synthetic_game import (
    EXPLORED, PHASES, PhaseThresholds, PositionSampler, SamplingConfig,
    classify_phase, iter_trajectory, policy_identity, simulate,
)


def position(white, player=game.WHITE, bar=0):
    board = [0] * 24
    for point, count in white.items():
        board[point] = count
    board[23] = -15
    return game.State(board, bar, 0, player)


def mirror(state):
    return game.State(-state.board[::-1], state.brown_bar, state.white_bar, -int(state.player))


class Phases(unittest.TestCase):
    def test_both_directions_and_precedence(self):
        cases = [
            (game.starting_state(), "early"),
            (position({3: 4, 16: 11}), "middle"),
            (position({11: 6, 16: 9}), "middle"),  # population large; distance short
            (position({3: 2, 22: 13}), "late"),
            (position({10: 1, 15: 14}), "late"),  # no leader within six pips
            (position({22: 14}, bar=1), "late"),
            (position({10: 1, 22: 14}), "near_terminal"),
            (position({14: 15}), "near_terminal"),  # eligible, including far leaders
        ]
        for state, expected in cases:
            for view in (state, mirror(state)):
                with self.subTest(expected=expected, view=game.state_key(view).hex()):
                    self.assertEqual(PHASES[classify_phase(view)], expected)

    def test_thresholds_and_serializable_config(self):
        state = position({10: 1, 22: 14})
        self.assertEqual(classify_phase(state, {"near_distance": 1}), 2)
        self.assertEqual(classify_phase(state, PhaseThresholds(near_lead=0)), 2)
        config = SamplingConfig(positions_per_game=11,
                                phase_weights=dict(zip(PHASES, (2, 3, 3, 2))),
                                phase_thresholds={"near_distance": 1})
        self.assertEqual(SamplingConfig(**asdict(config)), config)
        for bad in ({"phase_weights": (0, 0, 0, 0)}, {"phase_weights": (0, 1, 2)},
                    {"positions_per_game": True}, {"min_turn_gap": 0},
                    {"phase_thresholds": {"late_outside": 0}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SamplingConfig(**bad)


class Sampling(unittest.TestCase):
    @staticmethod
    def varied_states():
        # Many distinct full-count positions for both actors in every phase.
        for j in range(40):
            source, destination = 3 + j % 5, 15 + j // 5
            for phase in range(4):
                for color in (game.WHITE, game.BROWN):
                    if phase < 3:
                        count = (7, 4, 2)[phase]
                        yield position({source: count, destination: 15 - count}, color)
                    else:
                        yield position({12 + j % 3: 1 + j // 7,
                                        16 + j % 7: 14 - j // 7}, color)

    def test_bounded_unique_reservoir_phase_quotas_color_and_spacing(self):
        def sample(seed):
            config = SamplingConfig(positions_per_game=10)
            sampler = PositionSampler(seed, config)
            for number, state in enumerate(self.varied_states(), start=2):
                sampler.offer(number, state)
            self.assertLessEqual(sum(len(kept) for kept, _ in sampler.buckets.values()), 16 * 10)
            rows = sampler.select()
            self.assertEqual(rows, sampler.select())
            return rows
        rows = sample(11)
        self.assertEqual(rows, sample(11))
        self.assertNotEqual(rows, sample(12))
        self.assertEqual(Counter(r.phase for r in rows), {0: 2, 1: 3, 2: 3, 3: 2})
        self.assertEqual(Counter(r.player for r in rows), {1: 5, -1: 5})
        self.assertEqual(len({r.key for r in rows}), 10)
        self.assertTrue(all(b.turn - a.turn >= 3 for a, b in zip(rows, rows[1:])))

    def test_short_or_repeating_game_never_gets_manufactured_examples(self):
        sampler = PositionSampler(5, SamplingConfig())
        white = position({10: 1, 22: 14})
        brown = game.State(white.board, 0, 0, game.BROWN)
        for i in range(100):
            sampler.offer(2 + i, white if i % 2 == 0 else brown)
        selected = sampler.select()
        self.assertEqual([(r.turn, r.player) for r in selected], [(2, 1), (3, -1)])
        self.assertEqual(sampler.phase_visits, [0, 0, 0, 100])


class Fixed:
    """Real registered agents with a declared identity, for independent simulator comparisons."""
    def __init__(self, name):
        self.agent = create_agent(name)
        self.name = self.selected_strategy = name
        self.metadata = {"name": name}
        self.explored = False

    def choose_candidate(self, state, dice, rng=None):
        return self.agent.choose_candidate(state, dice, rng=rng)


class Simulation(unittest.TestCase):
    def attempt(self, cap=50, archived=True, sampling_seed=123, samples=8):
        spec = SimpleNamespace(dice_seed=7, sampling_seed=sampling_seed, game_id="fixture")
        with patch("synthetic_plan.build_policies", return_value=(Fixed("sprinter"), Fixed("sprinter"))):
            return simulate(spec, {"max_turns": cap}, SamplingConfig(positions_per_game=samples), archived)

    def test_every_archived_ply_replays_and_every_value_state_is_preroll(self):
        run = self.attempt(samples=50)
        incumbent = play_game("sprinter", "sprinter", 7, max_turns=50)
        self.assertEqual((run.winner, run.turns), (incumbent.winner, incumbent.turns))
        self.assertIsNotNone(run.winner)
        self.assertEqual(run.stats["white"], dict(incumbent.white_stats))
        self.assertEqual(run.stats["brown"], dict(incumbent.brown_stats))
        plies = list(iter_trajectory(run.trajectory))
        self.assertEqual(len(plies), run.turns)
        self.assertEqual(game.state_key(plies[0].state), run.initial)
        self.assertEqual(game.state_key(plies[-1].after), run.final)
        self.assertEqual(game.winner(plies[-1].after), run.winner)
        for ply in plies:
            self.assertEqual(game.apply_turn(ply.state, ply.dice, ply.moves), ply.after)
            candidate = evaluate_chosen(ply.state, game.Turn(ply.moves, ply.after))
            self.assertEqual((ply.captures, ply.self_captures), (candidate.captures, candidate.self_captures))
            self.assertEqual(policy_identity(ply.policy_code), ("sprinter", False))
            self.assertEqual(run.policy_trace[ply.number - 1], ply.policy_code)
        eligible_states = {game.state_key(ply.state) for ply in plies[1:]}
        self.assertEqual(len(run.samples), len(eligible_states))
        self.assertEqual({row[1] for row in run.samples}, eligible_states)
        for number, key, phase, code in run.samples:
            self.assertGreaterEqual(number, 2)
            self.assertEqual(key, game.state_key(plies[number - 1].state))
            self.assertEqual(phase, classify_phase(plies[number - 1].state))
            self.assertEqual(code, run.policy_trace[number - 1])
        self.assertEqual(sum(run.stats["phase_visits"]), run.turns - 1)

    def test_changing_sampling_or_archiving_preserves_entire_game(self):
        archived = self.attempt()
        compact = self.attempt(archived=False)
        changed = self.attempt(archived=True, sampling_seed=222, samples=2)
        self.assertIsNone(compact.trajectory)
        self.assertEqual(compact.samples, archived.samples)
        self.assertEqual(archived.trajectory, changed.trajectory)
        self.assertEqual(archived.final, compact.final)
        self.assertEqual(archived.policy_trace, compact.policy_trace)
        self.assertEqual(len(changed.samples), 2)

    def test_cap_and_opening_have_no_labeled_positions(self):
        capped = self.attempt(cap=4)
        self.assertIsNone(capped.winner)
        self.assertEqual(capped.samples, [])
        self.assertEqual(len(capped.policy_trace), 4)
        self.assertEqual(len(list(iter_trajectory(capped.trajectory))), 4)
        self.assertEqual(sum(capped.stats["phase_visits"]), 3)
        opening_only = self.attempt(cap=1)
        self.assertEqual(opening_only.stats["phase_visits"], [0] * 4)
        self.assertEqual(opening_only.samples, [])

    def test_board_is_offered_before_the_next_random_draw(self):
        events = []
        offer = PositionSampler.offer
        roll = game.roll_dice

        def watch_offer(sampler, number, state):
            events.append(number)
            return offer(sampler, number, state)

        def watch_roll(rng):
            events.append("draw")
            return roll(rng)

        with patch.object(PositionSampler, "offer", watch_offer), patch.object(game, "roll_dice", watch_roll):
            self.attempt(cap=4)
        self.assertEqual(events[events.index(2):], [2, "draw", 3, "draw", 4, "draw"])

    def test_reject_damaged_trajectory_and_unknown_policy(self):
        blob = self.attempt(cap=1).trajectory
        raw = zlib.decompress(blob)
        for corrupt in (blob[:-1], blob + b"x", zlib.compress(raw[:-1]),
                        zlib.compress(raw[:8] + b"\x02" + raw[9:])):
            with self.assertRaises(ValueError):
                list(iter_trajectory(corrupt))
        with self.assertRaises(ValueError):
            policy_identity(16)
        self.assertEqual(policy_identity(EXPLORED | AGENT_NAMES.index("random")), ("random", True))

    def test_plan_mixture_replays_the_same_choices_without_trajectory_recording(self):
        from synthetic_plan import PlanConfig, make_plan, spec_for

        config = PlanConfig(target_games=80, heldout_games=4, max_turns=8)
        specs = (spec_for(slot, 0, config) for slot in make_plan(config))
        spec = next(s for s in specs if s.white.kind == "turn" and s.brown.kind == "turn")
        archived = simulate(spec, config, SamplingConfig(), record_trajectory=True)
        compact = simulate(spec, config.as_dict(), SamplingConfig(), record_trajectory=False)
        self.assertEqual((archived.final, archived.policy_trace, archived.samples),
                         (compact.final, compact.policy_trace, compact.samples))
        self.assertEqual([p.policy_code for p in iter_trajectory(archived.trajectory)],
                         list(compact.policy_trace))
        self.assertTrue(all(policy_identity(c)[0] != "random" for c in compact.policy_trace))
        self.assertEqual(compact.white_meta, archived.white_meta)


if __name__ == "__main__":
    unittest.main()
