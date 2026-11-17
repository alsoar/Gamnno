"""Persistent history, packaged game discovery, and replay frame checks."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from history_store import HistoryStore
from history_tab import replay_frames


class HistoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "data" / "history.sqlite"
        self.store = HistoryStore(self.path)
        self.sample = self.store.load("baseline-2095")

    def human_game(self, color="white"):
        game = deepcopy(self.sample)
        game.update(id=str(uuid4()), completed_at="2026-11-16T19:05:03+00:00")
        game["players"][color] = "human"
        return game

    def test_bundled_catalog_contains_per_color_metadata_and_loads_record_on_demand(self):
        with patch("history_store._read_bundled", side_effect=AssertionError("loaded frames during listing")):
            summaries = self.store.list_games()
        short = next(g for g in summaries if g["id"] == "baseline-2095")
        self.assertEqual((short["seed"], short["game_type"], short["winner"]), (2095, "bot-bot", 0))
        self.assertEqual(short["players"], {"white": "baseline", "brown": "baseline"})
        self.assertEqual((short["turn_count"], short["move_count"]), (13, 34))
        self.assertIsNone(short["completed_at"])
        self.assertNotIn("turns", short)
        self.assertFalse(self.store.save(self.sample))
        with self.assertRaises(KeyError):
            self.store.load("unknown-recording")

    def test_save_is_idempotent_persistent_and_preserves_color_assignment(self):
        white, brown = self.human_game(), self.human_game("brown")
        brown["completed_at"] = "2026-11-17T08:12:00Z"
        self.assertTrue(self.store.save(white))
        self.assertTrue(self.store.save(brown))
        self.assertFalse(self.store.save(white))
        reopened = HistoryStore(self.path)
        self.assertEqual(reopened.load(white["id"]), white)
        with patch("history_store.zlib.decompress", side_effect=AssertionError("decoded an unselected game")):
            summaries = reopened.list_games()
        self.assertEqual([row["id"] for row in summaries[:2]], [brown["id"], white["id"]])
        self.assertEqual([row["game_type"] for row in summaries[:2]], ["human-bot", "human-bot"])
        self.assertEqual(summaries[0]["players"]["brown"], "human")
        self.assertEqual(summaries[1]["players"]["white"], "human")
        with self.assertRaisesRegex(ValueError, "different completed game"):
            reopened.save(dict(white, seed=42))
        self.assertEqual(reopened.load(white["id"]), white)

    def test_concurrent_duplicate_saves_insert_once(self):
        row = self.human_game()
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertCountEqual(pool.map(self.store.save, [row] * 4), [True, False, False, False])
        self.assertEqual(sum(s["id"] == row["id"] for s in self.store.list_games()), 1)

    def test_named_opponents_persist_for_both_matchup_types(self):
        for roles, expected in (({"white": "hoarder", "brown": "human"}, "human-bot"),
                                ({"white": "optimizer", "brown": "saboteur"}, "bot-bot")):
            record = self.human_game()
            record["players"] = roles
            self.assertTrue(self.store.save(record))
            summary = next(s for s in self.store.list_games() if s["id"] == record["id"])
            self.assertEqual((summary["players"], summary["game_type"]), (roles, expected))

    def test_only_completed_consistent_recordings_can_be_added(self):
        game = self.human_game()
        unfinished = deepcopy(game)
        unfinished["turns"].pop()
        tampered = deepcopy(game)
        tampered["turns"][0]["boards"][0] = tampered["opening"]
        wrong_actor = deepcopy(game)
        wrong_actor["turns"][1]["player"] = wrong_actor["turns"][0]["player"]
        invalid = [
            (dict(game, winner=None), "completed"), (unfinished, "winner"),
            (dict(game, winner=1-game["winner"]), "winner"),
            (tampered, "frame"), (wrong_actor, "player"),
            (dict(game, completed_at="2026-11-16T12:00:00"), "UTC"),
            (dict(game, id="baseline-2095"), "bundled"),
        ]
        for record, error in invalid:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.store.save(record)
        with self.assertRaises(KeyError):
            self.store.load(game["id"])

    def test_replay_includes_opening_all_checker_moves_and_win(self):
        frames = replay_frames(self.sample)
        expected = [self.sample["opening"]]
        for turn in self.sample["turns"]:
            expected.extend(turn["boards"])
        self.assertEqual([f.board for f in frames], expected)
        self.assertEqual([f.before for f in frames[1:]], expected[:-1])
        self.assertEqual(frames[-1].board[26], [1, 0])
        self.assertEqual(frames[-1].move["target"], 26)

    def test_pass_is_a_distinct_frame_without_a_checker_move(self):
        board = self.sample["opening"]
        frames = replay_frames({"opening": board, "turns": [{"moves": [], "boards": []}]})
        self.assertEqual(len(frames), 2)
        self.assertIsNone(frames[1].move)
        self.assertIs(frames[1].board, frames[0].board)


if __name__ == "__main__":
    unittest.main()
