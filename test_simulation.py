import sqlite3
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import backgammon as game
from agents import AGENT_NAMES
import simulation as sim
from training_data import PositionLog, iter_positions, pack_sample


class TournamentTests(unittest.TestCase):
    def test_default_continues_beyond_300_and_stops_after_2000(self):
        # Isolate the turn clock from policy runtime. The scripted outcome
        # uses a real terminal state with one White checker already borne off.
        active = {p: game.starting_state(p) for p in (game.WHITE,game.BROWN)}
        points = [-15] + [0]*22 + [14]
        terminal = game.State(points,0,0,game.BROWN)

        class Clock:
            name = "clock"
            def __init__(self, winning_turn):
                self.winning_turn = winning_turn
                self.calls = 0

            def choose_candidate(self, state, dice, rng):
                self.calls += 1
                next_state = terminal if self.calls == self.winning_turn else active[-int(state.player)]
                return SimpleNamespace(turn=game.Turn((),next_state))

        for stop,expected in ((301,game.WHITE),(2000,game.WHITE),(2001,None)):
            with self.subTest(stop=stop), patch.object(sim,"_observe"):
                actor = Clock(stop)
                result = sim.play_game(actor,actor)
                self.assertEqual(result.turns,min(stop,2000))
                self.assertEqual(actor.calls,result.turns)
                self.assertEqual(result.winner,expected)

    def test_all_45_pairings_self_play_seat_pool_and_behavior_denominators(self):
        report = sim.run_round_robin(AGENT_NAMES, 1, seed=61, max_turns=1)
        self.assertEqual((report["pairings"], report["games"], report["unfinished"]), (45,45,45))
        self.assertEqual(report["length"]["all_observed"]["histogram"], {1:45})
        self.assertEqual(report["length"]["completed"]["median"], None)
        for name in AGENT_NAMES:
            profile = report["profiles"][name]
            self.assertEqual(profile["appearances"], 10)
            self.assertEqual(profile["unfinished"], 10)
            self.assertEqual(len(report["matrix"][name]), 9)
            diagonal = report["matrix"][name][name]
            self.assertEqual((diagonal["games"], diagonal["appearances"]), (1,2))
            self.assertIsNone(diagonal["win_fraction_completed"])
            if profile["actor_turns"]:
                rates = profile["action_turn_fractions"]
                self.assertAlmostEqual(sum(rates[k] for k in ("pass","only_forward","only_backward","mixed")), 1)
        diversity = report["diversity"]
        self.assertEqual(len(diversity["requested_comparisons"]),5)
        self.assertEqual(len(diversity["pairs_by_similarity"]),36)

    def test_color_pair_and_compact_sample_replay(self):
        with TemporaryDirectory() as temp:
            db = Path(temp)/"examples.sqlite"
            report = sim.run_matchup("random", "sprinter", 2, seed=219, max_turns=4, positions=db)
            samples = list(iter_positions(db))
            self.assertEqual(len(samples),8)
            a,b = samples[:4],samples[4:]
            self.assertEqual((a[0].white,b[0].brown),("random","random"))
            self.assertEqual(a[0].seed,b[0].seed)
            self.assertNotEqual(a[0].game_id,b[0].game_id)
            self.assertEqual([s.example.dice for s in a],[s.example.dice for s in b])
            self.assertEqual([s.example.state.player for s in a],[s.example.state.player for s in b])
            for group in (a,b):
                state = group[0].example.state
                for ix,s in enumerate(group,start=1):
                    self.assertEqual(s.turn_number,ix)
                    self.assertEqual(s.example.state,state)
                    self.assertIsNone(s.example.winner)
                    self.assertEqual(s.acting_agent,s.white if state.player==game.WHITE else s.brown)
                    self.assertIn("stack_concentration", s.features["own_after"])
                    state = game.apply_turn(state,s.example.dice,s.example.action)
            with sqlite3.connect(db) as connection:
                sizes = [r[0] for r in connection.execute("SELECT length(example) FROM positions")]
            self.assertTrue(all(29<=size<=37 for size in sizes))
            self.assertEqual(report["profiles"]["random"]["actor_turns"],4)
            stats = report["profiles"]["random"]
            raw = [s.features for s in samples if s.acting_agent=="random"]
            self.assertEqual(stats["captures_per_game"], sum(f["captures"] for f in raw)/2)
            self.assertEqual(stats["mean_bar"], sum(f["own_after"]["bar"] for f in raw)/4)
            self.assertEqual(stats["forward_pips_per_turn"], sum(f["forward_progress"] for f in raw)/4)
            agreement = sim.decision_agreement(db, limit=1)
            self.assertEqual(agreement["positions_compared"], 1)
            self.assertEqual(len(agreement["pairs"]), len(sim.FOCUS_PAIRS))
            for pair in agreement["pairs"]:
                self.assertLessEqual(pair["same_moves"], pair["same_final_board"])
                self.assertIn(pair["same_final_board"], (0, 1))

    def test_sampling_does_not_perturb_play_and_disabled_never_packs(self):
        with patch.object(sim,"pack_sample",side_effect=AssertionError("unrequested sample")):
            baseline = sim.play_game("random","random",seed=91,max_turns=12)
        with TemporaryDirectory() as temp:
            first,second = Path(temp)/"all.db",Path(temp)/"few.db"
            full = sim.play_game("random","random",91,max_turns=12,positions=first)
            subset = sim.play_game("random","random",91,max_turns=12,positions=second,sample_every=3)
            self.assertEqual((full.white_stats,full.brown_stats),(baseline.white_stats,baseline.brown_stats))
            self.assertEqual((subset.white_stats,subset.brown_stats),(baseline.white_stats,baseline.brown_stats))
            all_rows = {p.turn_number:p.example for p in iter_positions(first)}
            few = list(iter_positions(second))
            self.assertGreater(len(few),0)
            self.assertLess(len(few),len(all_rows))
            self.assertTrue(all(s.example==all_rows[s.turn_number] for s in few))

    def test_agreement_deduplicates_rolls_balances_actors_and_fills_short_strata(self):
        state = game.starting_state(game.WHITE)
        blocked = game.State([-2, -2] + [0] * 20 + [-11, 14], 1, 0, game.WHITE)
        self.assertEqual(game.legal_turns(blocked, (1, 2))[0].moves, ())
        examples = {
            "baseline": [(state, (1, 2)), (blocked, (1, 2))],
            "hoarder": [(state, r) for r in ((2, 1), (1, 3), (1, 4), (1, 5), (1, 6))],
            "sprinter": [(state, r) for r in ((2, 3), (2, 4), (2, 5), (2, 6))],
        }
        with TemporaryDirectory() as temp:
            path = Path(temp) / "positions.db"
            with PositionLog(path) as store:
                for name, positions in examples.items():
                    samples = [pack_sample(i, s, r, game.legal_turns(s, r)[0], {})
                               for i, (s, r) in enumerate(positions, 1)]
                    store.write(sim.GameResult(name, 7, name, "random", None,
                                               len(samples), Counter(), Counter(), samples))
            result = sim.decision_agreement(path, 6, seed=506)
            self.assertEqual(result, sim.decision_agreement(path, 6, seed=506))
            self.assertEqual(result["logged_rows_examined"], 11)
            self.assertEqual(result["repeated_position_roll_skipped"], 1)
            self.assertEqual(result["unique_inputs_by_source"], {"baseline": 2, "hoarder": 4, "sprinter": 4})
            self.assertEqual(result["source_actors"]["baseline"], 1)
            self.assertEqual(sorted(result["source_actors"].values()), [1, 2, 3])
            self.assertNotEqual(result["selection_sha256"], sim.decision_agreement(path, 6, seed=507)["selection_sha256"])
            all_inputs = sim.decision_agreement(path, 100, seed=506)
            self.assertEqual(all_inputs["positions_compared"], 9)
            self.assertEqual(all_inputs["forced_position_roll_skipped"], 1)
            self.assertEqual(all_inputs["source_actors"], {"baseline": 1, "hoarder": 4, "sprinter": 4})

    def test_win_at_cap_and_independent_selfplay_seeds(self):
        def ready(actor):
            board = [0] * 24
            board[0],board[23] = -15,15
            return game.State(board,0,0,actor)

        with TemporaryDirectory() as temp, patch.object(game,"starting_state",side_effect=ready):
            db = Path(temp)/"samples.db"
            result = sim.run_matchup("sprinter","sprinter",2,seed=99,max_turns=1,positions=db)
            self.assertEqual(result["completed"],2)
            diagonal = result["matrix"]["sprinter"]["sprinter"]
            self.assertEqual((diagonal["wins"],diagonal["losses"],diagonal["win_fraction_all"]),(2,2,.5))
            self.assertEqual(result["profiles"]["sprinter"]["action_counts"]["bear_off"],2)
            rows = list(iter_positions(db))
            self.assertEqual([s.seed for s in rows],[99,100])
            for row in rows:
                self.assertEqual(row.example.winner,row.example.state.player)
                self.assertEqual(row.example.value_target,1)

    def test_finished_and_all_start_rates_with_odd_color_assignment(self):
        def result(job):
            a,b,i,seed,limit,_ = job
            a_white = i%2==0
            white,brown = (a,b) if a_white else (b,a)
            winner = [game.WHITE,game.WHITE,None,game.BROWN][i]
            return (a,b), a_white, sim.GameResult(str(i),seed,white,brown,winner,
                                                 limit if winner is None else 2+i,Counter(),Counter())
        with patch.object(sim,"_execute",side_effect=result):
            report = sim.run_matchup("baseline","sprinter",4,max_turns=10)
        a,b = report["matrix"]["baseline"]["sprinter"],report["matrix"]["sprinter"]["baseline"]
        self.assertEqual((a["wins"],a["losses"],a["unfinished"]),(2,1,1))
        self.assertEqual((a["win_fraction_all"],a["win_fraction_completed"]),(.5,2/3))
        self.assertEqual((b["win_fraction_all"],b["win_fraction_completed"]),(.25,1/3))
        self.assertEqual(report["length"]["completed"]["histogram"],{2:1,3:1,5:1})
        self.assertEqual(report["length"]["all_observed"]["median"],4)
        self.assertEqual(report["length"]["unfinished_observed"]["mean"],10)

    def test_matching_parallel_and_serial_reports(self):
        settings = dict(seed=889,max_turns=2)
        serial=sim.run_matchup("random","baseline",2,**settings)
        parallel=sim.run_matchup("random","baseline",2,workers=2,**settings)
        self.assertEqual(serial,parallel)

    def test_equal_profiles_flag_and_no_false_difference_from_constants(self):
        v = dict.fromkeys(sim.PROFILE_METRICS,1.)
        report = sim.compare_profiles({"fortress":v,"hoarder":v})
        self.assertEqual(report["requested_comparisons"][0]["distance"],0.)
        self.assertEqual(report["flagged_near_pairs"],[["fortress","hoarder"]])


if __name__ == "__main__":
    unittest.main()
