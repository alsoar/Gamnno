"""Exercise the real parallel generator, checkpoints, portable export and archive reader."""
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from synthetic_audit import AuditError, audit
from synthetic_data import exclusive, generate
from synthetic_game import SamplingConfig
from synthetic_plan import PlanConfig
from synthetic_store import GAME_COLUMNS, SPLITS, append_game, digest, evidence_db


def _run(root, plan, sampling, **kwargs):
    return generate(root, plan, sampling, trajectory="all", run_audit=False,
                    progress_seconds=3600, **kwargs)


def _logical_evidence(root):
    """The game decision, sample and archive bytes are reproducible; timings are measured."""
    columns = [key for key in GAME_COLUMNS if key not in ("elapsed", "peak_mib")]
    with evidence_db(root) as db:
        games = [tuple(r) for r in db.execute(
            "SELECT " + ",".join(columns) + " FROM games ORDER BY slot_id,attempt")]
        samples = [tuple(r) for r in db.execute(
            "SELECT game_id,turn_number,state,phase,policy FROM samples ORDER BY game_id,turn_number")]
    return games, samples


def _write_competitor(root, result):
    try:
        _run(root, PlanConfig(target_games=0, heldout_games=0), SamplingConfig(), workers=1)
    except Exception as error:
        result.put((type(error).__name__, str(error)))
    else:
        result.put(("success", ""))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        # This deterministic 2-slot case includes a genuine 14-turn finish on
        # the second attempt of one slot, and three genuinely capped attempts.
        cls.plan = PlanConfig(target_games=2, heldout_games=0, namespace="integration",
                              max_turns=30, max_attempts_per_slot=2,
                              mixture_weights={"sprinter": 1})
        cls.sampling = SamplingConfig(positions_per_game=4)
        cls.reference = cls.root / "reference"
        cls.ref_manifest = _run(cls.reference, cls.plan, cls.sampling, workers=2)

    def test_budgeted_restart_change_workers_and_raise_slot_cap(self):
        root = self.root / "resume"
        low = replace(self.plan, max_attempts_per_slot=1)
        first = _run(root, low, self.sampling, workers=1, attempt_budget=1)
        self.assertEqual(first["attempted_games"], 1)
        initial_game = _logical_evidence(root)[0][0]
        retry_limit = _run(root, low, self.sampling, workers=2, attempt_budget=20)
        self.assertEqual((retry_limit["attempted_games"], retry_limit["unfinished_games"]), (2, 2))
        self.assertEqual(retry_limit["sampled_positions"], 0)

        final = _run(root, self.plan, self.sampling, workers=1)
        for field in ("attempted_games", "unfinished_games", "completed_games", "sampled_positions",
                      "split_sizes", "split_games", "cross_split_drops"):
            self.assertEqual(final[field], self.ref_manifest[field])
        self.assertEqual((final["attempted_games"], final["unfinished_games"],
                          final["completed_games"], final["sampled_positions"]), (4, 3, 1, 4))
        games, samples = _logical_evidence(root)
        self.assertEqual((games, samples), _logical_evidence(self.reference))
        self.assertIn(initial_game, games)
        self.assertEqual(len({r[0] for r in games}), 4)
        self.assertEqual(len({(r[1], r[2]) for r in games}), 4)
        self.assertEqual(len({(r[0], r[1]) for r in samples}), 4)
        eligible = {row[0] for row in games if row[GAME_COLUMNS.index("winner")] is not None}
        self.assertEqual({row[0] for row in samples}, eligible)
        self.assertEqual(final["config"]["plan"]["max_attempts_per_slot"], 2)
        self.assertEqual(final["config"]["created_utc"], first["config"]["created_utc"])
        for name, checksum in final["checksums"].items():
            if name.split("/")[0] in (*SPLITS, "trajectories") or name == "agent_metadata.json":
                self.assertEqual(checksum, self.ref_manifest["checksums"][name], name)

        # All slots are either filled or exhausted. No workers or attempts are needed.
        with patch("synthetic_data.ProcessPoolExecutor", side_effect=AssertionError("unexpected resimulation")):
            again = _run(root, self.plan, self.sampling, workers=3)
        self.assertEqual(again["attempted_games"], 4)
        self.assertEqual(_logical_evidence(root), (games, samples))

    def test_interrupted_checkpoint_rolls_back_samples_and_can_be_retried(self):
        root = self.root / "interrupted"
        # Use one worker and one-attempt budgets so committed effects are observable
        # on both sides of a failure after inserting the winning game's sample rows.
        _run(root, self.plan, self.sampling, workers=1, attempt_budget=3)
        before = _logical_evidence(root)
        self.assertEqual((len(before[0]), len(before[1])), (3, 0))
        def interrupt(db, spec, result, elapsed, peak):
            append_game(db, spec, result, elapsed, peak)
            self.assertIsNotNone(result.winner)
            self.assertGreater(len(result.samples), 0)
            raise RuntimeError("test interruption after game and sample inserts")
        with patch("synthetic_data.append_game", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "test interruption"):
                _run(root, self.plan, self.sampling, workers=1, attempt_budget=1)
        self.assertEqual(_logical_evidence(root), before)
        self.assertEqual(json.loads((root / "manifest.json").read_text())["attempted_games"], 3)
        _run(root, self.plan, self.sampling, workers=1, attempt_budget=1)
        self.assertEqual(_logical_evidence(root), _logical_evidence(self.reference))

    def test_conflicting_configuration_preserves_existing_files(self):
        root = self.root / "identity"
        _run(root, self.plan, self.sampling, workers=1, attempt_budget=0)
        paths = [root / n for n in ("manifest.json", "config.json", "agent_metadata.json",
                                     "checkpoints/run.sqlite3")]
        before = {f: digest(f) for f in paths}
        for plan, sampling in [(replace(self.plan, master_seed=999), self.sampling),
                               (self.plan, SamplingConfig(positions_per_game=5)),
                               (replace(self.plan, max_turns=31), self.sampling)]:
            with self.subTest(plan=plan, sampling=sampling), self.assertRaisesRegex(ValueError, "identity"):
                _run(root, plan, sampling, workers=1, attempt_budget=0)
            self.assertEqual({f: digest(f) for f in paths}, before)
        with self.assertRaisesRegex(ValueError, "attempt ceiling"):
            _run(root, replace(self.plan, max_attempts_per_slot=1), self.sampling,
                 workers=1, attempt_budget=0)
        self.assertEqual({f: digest(f) for f in paths}, before)

    def test_interrupted_shard_export_recovers_from_checkpoint_without_new_games(self):
        root = self.root / "interrupted-export"
        initial = _run(root, self.plan, self.sampling, workers=1, attempt_budget=3)
        manifest_before = (root / "manifest.json").read_bytes()
        game_shard = root / "games/part-00000.parquet"
        checkpoint = root / "checkpoints/run.sqlite3"
        original_replace = Path.replace
        interrupted = []

        def interrupt_after_game_shard(source, target):
            target = Path(target)
            staged_digest = digest(source) if target == game_shard else None
            result = original_replace(source, target)
            if target == game_shard:
                # The replacement publishes one complete file before the later
                # value/trajectory shards and final manifest can be published.
                self.assertFalse(source.exists())
                self.assertEqual(digest(target), staged_digest)
                interrupted.append(digest(checkpoint))
                raise RuntimeError("test interruption after replacing game shard")
            return result

        with patch.object(Path, "replace", interrupt_after_game_shard):
            with self.assertRaisesRegex(RuntimeError, "after replacing game shard"):
                _run(root, self.plan, self.sampling, workers=1, attempt_budget=1)
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(digest(checkpoint), interrupted[0])
        self.assertEqual((root / "manifest.json").read_bytes(), manifest_before)
        self.assertNotEqual(digest(game_shard), initial["checksums"]["games/part-00000.parquet"])
        self.assertEqual(_logical_evidence(root), _logical_evidence(self.reference))
        with self.assertRaisesRegex(AuditError, "checksum differs"):
            audit(root, write_report=False)
        self.assertEqual(digest(checkpoint), interrupted[0])

        # Equivalent to resuming the original command with --attempt-budget 0.
        with patch("synthetic_data.ProcessPoolExecutor", side_effect=AssertionError("unexpected resimulation")):
            final = _run(root, self.plan, self.sampling, workers=2, attempt_budget=0)
        self.assertEqual(digest(checkpoint), interrupted[0])
        self.assertEqual(final["attempted_games"], 4)
        self.assertEqual(_logical_evidence(root), _logical_evidence(self.reference))
        for relative, checksum in final["checksums"].items():
            self.assertEqual(digest(root / relative), checksum)
            if relative.split("/")[0] in (*SPLITS, "trajectories"):
                self.assertEqual(checksum, self.ref_manifest["checksums"][relative])
        report = audit(root, write_report=False)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["validation"]["trajectory_games"], 4)
        self.assertEqual(report["validation"]["trajectory_checked_samples"], 4)

    def test_another_process_cannot_write_while_lock_held(self):
        root = self.root / "write-lock"
        (root / "checkpoints").mkdir(parents=True)
        ctx = multiprocessing.get_context("spawn")
        result = ctx.Queue()
        try:
            with exclusive(root):
                process = ctx.Process(target=_write_competitor, args=(root, result))
                process.start()
                try:
                    kind, message = result.get(timeout=30)
                    self.assertEqual(kind, "RuntimeError")
                    self.assertIn("another generator", message)
                    process.join(timeout=5)
                    self.assertEqual(process.exitcode, 0)
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join(5)
            self.assertFalse((root / "config.json").exists())
            fresh = _run(root, self.plan, self.sampling, workers=1, attempt_budget=0)
            self.assertEqual(fresh["attempted_games"], 0)
        finally:
            result.close()
            result.join_thread()

    def test_portable_archive_reconstructs_and_validates_without_checkpoint(self):
        archive = self.root / "portable"
        for relative in ("manifest.json", *self.ref_manifest["checksums"]):
            out = archive / relative
            out.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(self.reference / relative, out)
        self.assertFalse((archive / "checkpoints").exists())
        self.assertEqual(_logical_evidence(archive), _logical_evidence(self.reference))
        checks = audit(archive, write_report=False)
        self.assertEqual(checks["status"], "passed")
        self.assertFalse(checks["validation"]["has_prefilter_checkpoint"])
        self.assertEqual(checks["validation"]["trajectory_games"], 4)
        self.assertEqual(checks["validation"]["trajectory_checked_samples"], 4)
        self.assertFalse((archive / "checkpoints").exists())
        before = {name: digest(archive / name) for name in ("manifest.json", *self.ref_manifest["checksums"])}
        with self.assertRaisesRegex(ValueError, "no resume checkpoint"):
            _run(archive, self.plan, self.sampling, workers=1, attempt_budget=0)
        self.assertFalse((archive / "checkpoints/run.sqlite3").exists())
        self.assertEqual({name: digest(archive / name) for name in before}, before)


if __name__ == "__main__":
    unittest.main()
