"""Offline integrity checks and empirical statistics for the synthetic value dataset."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
from random import Random
import sqlite3
import struct
from time import perf_counter

import backgammon as game


class AuditError(ValueError):
    """The dataset is internally inconsistent."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def distribution(histogram):
    """Counts, mean, and linearly interpolated empirical percentiles, storing only bins."""
    bins = sorted(histogram.items())
    size = sum(count for _, count in bins)

    def percentile(p):
        if not size:
            return None
        rank = p * (size - 1)
        lo, hi = math.floor(rank), math.ceil(rank)
        cumulative = 0
        for value, count in bins:
            if cumulative <= lo < cumulative + count:
                lower = value
            if cumulative <= hi < cumulative + count:
                return lower + (rank - lo) * (value - lower)
            cumulative += count

    return {"count": size, "mean": _rate(sum(value * n for value, n in bins), size),
            "p10": percentile(.10), "median": percentile(.50), "p90": percentile(.90),
            "p95": percentile(.95), "p99": percentile(.99),
            "min": bins[0][0] if bins else None, "max": bins[-1][0] if bins else None,
            "histogram": dict(bins)}


def validate_wire(wire, *, terminal_allowed=False):
    """Independently check the signed encoding and conservation before constructing a State."""
    require(isinstance(wire, bytes) and len(wire) == 27, "expected a 27-byte position")
    fields = struct.unpack("24bBBb", wire)
    board, white_bar, brown_bar, actor = fields[:24], *fields[24:]
    require(actor in (1, -1), "invalid player to move")
    require(all(-15 <= p <= 15 for p in board) and max(white_bar, brown_bar) <= 15,
            "invalid point or bar count")
    white_off = 15 - white_bar - sum(max(0, p) for p in board)
    brown_off = 15 - brown_bar - sum(max(0, -p) for p in board)
    require((white_off, brown_off) in ((0, 0), (1, 0), (0, 1)), "checker conservation failure")
    require(terminal_allowed or white_off + brown_off == 0, "terminal state in pre-roll position")
    return game.deserialize_state(wire)


def _actor_key(wire):
    """Exact pre-roll value input, re-encoded with the side to move as White."""
    if wire[-1] == 1:
        return wire
    points = struct.unpack("24b",wire[:24])
    return struct.pack("24bBBb", *(-p for p in reversed(points)), wire[25], wire[24], 1)


class StateIndex:
    """Keep exact, collision-free board counts on disk, including within-split repeats."""

    def __init__(self, database):
        self.db = database
        database.execute("""CREATE TABLE unique_states (
            state BLOB PRIMARY KEY, split TEXT NOT NULL, occurrences INTEGER NOT NULL
        ) WITHOUT ROWID""")
        database.execute("""CREATE TABLE model_inputs (
            state BLOB PRIMARY KEY, split TEXT NOT NULL, occurrences INTEGER NOT NULL
        ) WITHOUT ROWID""")

    def add(self, state, split):
        changed = self.db.execute("""INSERT INTO unique_states VALUES(?,?,1)
            ON CONFLICT(state) DO UPDATE SET occurrences=occurrences+1 WHERE split=excluded.split""",
                                  (state, split)).rowcount
        require(changed == 1, "identical board and active color found in different splits")
        changed = self.db.execute("""INSERT INTO model_inputs VALUES(?,?,1)
            ON CONFLICT(state) DO UPDATE SET occurrences=occurrences+1 WHERE split=excluded.split""",
                                  (_actor_key(state),split)).rowcount
        require(changed == 1, "identical player-normalized model input found in different splits")

    def report(self):
        rows = self.db.execute("""SELECT split,COUNT(*),SUM(occurrences),MAX(occurrences)
            FROM unique_states GROUP BY split""")
        by_split = {split: {"unique": unique, "positions": total, "max_visits": most,
                            "repeat_fraction": (total - unique) / total}
                    for split, unique, total, most in rows}
        unique = sum(s["unique"] for s in by_split.values())
        total = sum(s["positions"] for s in by_split.values())
        normalized = self.db.execute("SELECT COUNT(*) FROM model_inputs").fetchone()[0]
        return {"unique": unique, "positions": total, "repeat_count": total - unique,
                "repeat_fraction": _rate(total - unique, total), "by_split": by_split,
                "unique_model_inputs": normalized, "normalized_repeat_count": total-normalized,
                "normalized_repeat_fraction":_rate(total-normalized,total)}


SPLITS = ("train", "validation", "test", "heldout_evaluation")
PHASES = ("early", "middle", "late", "near_terminal")
VALUE_COLUMNS = frozenset((
    "game_id", "position_id", "turn_number", "board", "white_bar", "brown_bar", "player_to_move",
    "winner", "win_target", "white_agent", "brown_agent", "matchup_category", "mixture_type",
    "game_phase", "game_length", "generation_seed", "source_policy_metadata",
))


def outcomes(counter):
    n, w, l = counter["attempts"], counter["wins"], counter["losses"]
    return {"attempted": n, "completed": w+l, "unfinished": n-w-l, "wins": w, "losses": l,
            "completion_rate": _rate(w+l, n), "win_rate_completed": _rate(w, w+l),
            "win_rate_all": _rate(w, n)}


class Cohort:
    def __init__(self):
        self.seats = defaultdict(Counter)
        self.matchups = defaultdict(Counter)
        self.categories = defaultdict(Counter)
        self.subcategories = defaultdict(Counter)
        self.splits = defaultdict(Counter)
        self.behavior = Counter()
        self.lengths = Counter()
        self.finished_lengths = Counter()
        self.reasons = Counter()
        self.moves = defaultdict(Counter)
        self.opener = Counter()
        self.visits = Counter()

    def add(self, entry, stats):
        n, win = entry["turns"], entry["winner"]
        category = self.categories[entry["category"]]
        subcategory = self.subcategories[entry["subcategory"]]
        split = self.splits[entry["split"]]
        matchup = self.matchups[(entry["white_agent"],entry["brown_agent"])]
        for group in (category, subcategory, split, matchup, self.opener):
            group["attempts"] += 1
            group["total_turns"] += n
        self.lengths[n] += 1
        if win is None:
            self.reasons[entry["reason"]] += 1
        else:
            self.finished_lengths[n] += 1
            # Category and color-specific matchup use white as the reference color.
            for group in (category, subcategory, split, matchup):
                group["wins" if win == game.WHITE else "losses"] += 1
            self.opener["wins" if win == entry["opener"] else "losses"] += 1
        for side, color in (("white", game.WHITE), ("brown", game.BROWN)):
            seat = self.seats[entry[side + "_agent"]]
            seat["attempts"] += 1
            seat["total_turns"] += n
            if win is not None:
                seat["wins" if win == color else "losses"] += 1
            self.behavior.update(stats[side])
        for key in ("captures", "self_captures", "forward_pips", "backward_pips", "backward_steps"):
            value = stats["white"].get(key, 0) + stats["brown"].get(key, 0)
            self.moves[key][value] += 1
        self.visits.update(dict(zip(PHASES,stats["phase_visits"])))

    @staticmethod
    def _group(counter):
        return dict(outcomes(counter), mean_observed_turns=_rate(counter["total_turns"],counter["attempts"]))

    def report(self):
        n = sum(self.lengths.values())
        b = self.behavior
        return {
            "first_player": outcomes(self.opener),
            "all_observed_turns": distribution(self.lengths),
            "completed_game_turns": distribution(self.finished_lengths),
            "unfinished_reasons": dict(self.reasons),
            "by_agent": {k:self._group(v) for k,v in sorted(self.seats.items())},
            "by_matchup": [{"white":w,"brown":b,**self._group(v)} for (w,b),v in sorted(self.matchups.items())],
            "by_category": {k:self._group(v) for k,v in sorted(self.categories.items())},
            "by_subcategory": {k:self._group(v) for k,v in sorted(self.subcategories.items())},
            "by_split": {k:self._group(v) for k,v in sorted(self.splits.items())},
            "phase_visits_before_sampling": dict(self.visits),
            "per_attempt_distributions": {k:distribution(v) for k,v in self.moves.items()},
            "behavior": {
                "total_actor_turns": b["own_turns"], "total_checker_steps": b["checker_moves"],
                "captures_per_attempt": _rate(b["captures"],n),
                "self_captures_per_attempt": _rate(b["self_captures"],n),
                "forward_pips_per_attempt": _rate(b["forward_pips"],n),
                "backward_pips_per_attempt": _rate(b["backward_pips"],n),
                "backward_step_fraction": _rate(b["backward_steps"],b["checker_moves"]),
                "mean_blots": _rate(b["blots"],b["own_turns"]),
                "mean_bar": _rate(b["bar"],b["own_turns"]),
                "mean_concentration": _rate(b["concentration"],b["own_turns"]),
                "mean_blockade": _rate(b["blockade"],b["own_turns"]),
                "side_eligible_after_ply_fraction": _rate(b["eligible_plies"],b["board_plies"]),
            },
        }


def _capture_counts(state, moves):
    """Count hits stepwise, including a blot created or vacated earlier in this turn."""
    board = list(struct.unpack("24b", game.state_key(state)[:24]))
    hits = own_hits = 0
    for checker, source, target, _ in moves:
        if source != game.BAR:
            board[source] -= checker
        if target == game.OFF:
            continue
        if board[target] == -checker:
            if checker == state.player:
                hits += 1
            else:
                own_hits += 1
            board[target] = 0
        board[target] += checker
    return hits, own_hits


def _check_policies(spec, plan, entry):
    """Reproduce each seat's strategy/exploration draws without making game decisions."""
    from agents import AGENT_NAMES
    from synthetic_plan import POLICY_VERSION, child_seed

    for key, color in (("white",1),("brown",-1)):
        desc = getattr(spec,key)
        archived = json.loads(entry[key+"_meta"])
        require((archived["policy_id"],archived["kind"],archived["base"],archived["epsilon"],archived["agent_version"])
                == (desc.name,desc.kind,desc.base,desc.epsilon,POLICY_VERSION), "strategy wrapper metadata differs")
        seat_seed = child_seed(spec.generation_seed,key)
        select = Random(child_seed(seat_seed,"strategy-selection"))
        explore = Random(child_seed(seat_seed,"exploration-coin"))
        weights = plan.heldout_weights if desc.heldout else plan.mixture_weights
        fixed = desc.base
        if desc.kind in ("game","turn"):
            require(archived["mixture_weights"] == dict(zip(AGENT_NAMES,weights)), "mixture weights differ")
        if desc.kind == "game":
            fixed = select.choices(AGENT_NAMES,weights=weights,k=1)[0]
            require(fixed == archived["game_strategy"], "game mixture choice differs")
        for actual in entry["policy_trace"][int(color!=entry["opener"])::2]:
            name, explored = fixed, False
            if desc.kind == "turn":
                name = select.choices(AGENT_NAMES,weights=weights,k=1)[0]
            elif desc.kind == "noisy" and explore.random() < desc.epsilon:
                name, explored = "random", True
            expected = AGENT_NAMES.index(name) | (128 if explored else 0)
            require(expected == actual, "per-turn pure strategy or exploration flag differs from its seed")


def _check_trajectory(blob, entry, samples, spec, stats):
    from synthetic_game import iter_trajectory

    roller = Random(spec.dice_seed)
    opener, dice = game.opening_roll(roller)
    require(opener == entry["opener"], "seeded opening color differs")
    state = game.starting_state(opener)
    trace = entry["policy_trace"]
    count = matched = 0
    hits = {game.WHITE:Counter(),game.BROWN:Counter()}
    for count, ply in enumerate(iter_trajectory(blob), 1):
        require(count <= entry["turns"] and count == ply.number, "trajectory turn index differs")
        require(ply.state == state and tuple(ply.dice) == dice, "trajectory board chain or seeded dice differs")
        require(game.winner(state) is None, "action after first bear-off")
        require(trace[count-1] == ply.policy_code, "trajectory policy differs from policy trace")
        actual = game.apply_turn(state, dice, ply.moves)
        require(actual == ply.after, "recorded action does not yield recorded successor")
        c, own = _capture_counts(state, ply.moves)
        require((c, own) == (ply.captures, ply.self_captures), "trajectory hits differ")
        hits[int(state.player)].update({"captures": c, "self_captures": own})
        if count in samples:
            require(samples[count][0] == game.state_key(state), "sample not reachable on recorded turn")
            matched += 1
        state, dice = actual, game.roll_dice(roller)
    require(count == entry["turns"] and game.state_key(state) == entry["final"], "trajectory final state differs")
    require(matched == len(samples), "sample outside replayed trajectory")
    for color,key in ((game.WHITE,"white"),(game.BROWN,"brown")):
        for event in ("captures","self_captures"):
            require(hits[color][event] == stats[key].get(event,0), "aggregate capture evidence differs")
    return matched


def _read_games(db, thresholds, plans, sampling, verify_seeds):
    from synthetic_game import classify_phase, simulate
    cohorts = {"primary": Cohort(), "heldout_evaluation": Cohort()}
    coverage = Counter({k:0 for k in ("attempts","candidate_positions_examined","trajectory_checked_samples",
                                      "trajectory_games","trajectory_turns","regenerated_games","regenerated_samples",
                                      "games_without_offline_replay","samples_without_offline_replay")})
    for g in db.execute("SELECT * FROM games ORDER BY slot_id,attempt"):
        gid = g["game_id"]
        try:
            require(g["split"] in SPLITS, "unknown split")
            opener, winner, n = g["opener"], g["winner"], g["turns"]
            require(opener in (-1,1) and winner in (-1,1,None) and n >= 1, "invalid game result")
            require(game.state_key(game.starting_state(opener)) == g["initial"], "wrong starting position")
            final = validate_wire(g["final"], terminal_allowed=True)
            require(game.winner(final) == winner, "winner differs from terminal state")
            require(final.player == opener * (-1 if n % 2 else 1), "wrong player on final state")
            require(winner is not None or bool(g["reason"]), "unfinished attempt has no reason")
            trace = g["policy_trace"]
            require(len(trace) == n and all((value & 127) < 9 for value in trace), "invalid policy trace")
            stats = json.loads(g["stats"])
            require(len(stats["phase_visits"]) == 4 and sum(stats["phase_visits"]) == n - 1,
                    "phase observations do not count pre-roll turns")
            for color, side in ((game.WHITE,"white"),(game.BROWN,"brown")):
                require(stats[side].get("own_turns",0) == (n + int(color == opener)) // 2,
                        "incorrect actor-turn count")
            spec, plan_config = plans(g)
            require((spec.game_id, spec.generation_seed, spec.split, spec.category, spec.subcategory)
                    == (gid, g["generation_seed"], g["split"], g["category"], g["subcategory"]),
                    "seed, split, or category differs from deterministic allocation")
            require((spec.white_agent,spec.brown_agent,spec.mixture_type) ==
                    (g["white_agent"],g["brown_agent"],g["mixture_type"]), "allocated seat metadata differs")
            _check_policies(spec,plan_config,g)
            require(1 <= n <= plan_config.max_turns and (winner is not None or n == plan_config.max_turns),
                    "game duration inconsistent with turn limit")
            samples = {}
            for turn, wire, phase, policy in db.execute(
                    "SELECT turn_number,state,phase,policy FROM samples WHERE game_id=? ORDER BY turn_number", (gid,)):
                require(winner is not None and 2 <= turn <= n, "sample belongs to unfinished game or invalid turn")
                state = validate_wire(wire)
                require(state.player == opener * (1 if turn % 2 else -1), "incorrect sampled actor")
                require(0 <= phase <= 3 and classify_phase(state, thresholds) == phase, "sampled phase differs")
                require(trace[turn-1] == policy, "sampled policy differs from trace")
                samples[turn] = (wire, phase, policy)
            require(len(samples) <= sampling.positions_per_game, "per-game sample limit exceeded")
            require(len({s[0] for s in samples.values()}) == len(samples), "identical state sampled twice in one game")
            coverage["attempts"] += 1
            coverage["candidate_positions_examined"] += len(samples)
            if g["trajectory"] is not None:
                coverage["trajectory_checked_samples"] += _check_trajectory(g["trajectory"], g, samples, spec, stats)
                coverage["trajectory_games"] += 1
                coverage["trajectory_turns"] += n
            elif verify_seeds == -1 or coverage["regenerated_games"] < verify_seeds:
                regenerated = simulate(spec, plan_config, sampling, record_trajectory=True)
                require((regenerated.opener,regenerated.winner,regenerated.turns,regenerated.final,regenerated.policy_trace)
                        == (opener,winner,n,g["final"],trace), "seeded regeneration differs")
                generated_samples = {t:(w,p,c) for t,w,p,c in regenerated.samples}
                require(all(generated_samples.get(t) == s for t,s in samples.items()),
                        "regenerated sample differs")
                _check_trajectory(regenerated.trajectory,g,samples,spec,stats)
                coverage["regenerated_games"] += 1
                coverage["regenerated_samples"] += len(samples)
            else:
                coverage["games_without_offline_replay"] += 1
                coverage["samples_without_offline_replay"] += len(samples)
            cohort = "heldout_evaluation" if g["split"] == "heldout_evaluation" else "primary"
            cohorts[cohort].add(g, stats)
        except (ValueError, IndexError, KeyError, TypeError) as error:
            raise AuditError(f"{gid}: {error}") from error
    return {name:c.report() for name,c in cohorts.items()}, dict(coverage)


class PositionStats:
    def __init__(self):
        self.labels = Counter()
        self.colors = Counter()
        self.phases = Counter()
        self.actors = Counter()
        self.wrappers = Counter()
        self.matchups = Counter()
        self.categories = Counter()
        self.bars = Counter()
        self.concentration = Counter()
        self.remaining = Counter()

    def add(self, row, wire, actor, strategy, category, matchup):
        self.labels[row["win_target"]] += 1
        self.colors["white" if row["player_to_move"] == 1 else "brown"] += 1
        self.phases[PHASES[row["game_phase"]]] += 1
        self.actors[strategy] += 1
        self.wrappers[actor] += 1
        self.matchups[matchup] += 1
        self.categories[category] += 1
        self.bars[row["white_bar"]+row["brown_bar"]] += 1
        signed = struct.unpack("24b",wire[:24])
        self.concentration[sum(v*v for v in signed if v*row["player_to_move"] > 0)] += 1
        self.remaining[row["game_length"] - row["turn_number"] + 1] += 1

    def report(self):
        total = sum(self.labels.values())
        return {"count":total,"labels":dict(self.labels),"winning_label_fraction":_rate(self.labels[1],total),
                "colors":dict(self.colors),"phases":dict(self.phases),"selected_strategy":dict(self.actors),
                "acting_agent":dict(self.wrappers),"matchup":dict(self.matchups),"category":dict(self.categories),
                "bar_occupancy":distribution(self.bars),"actor_concentration":distribution(self.concentration),
                "turns_until_actual_win":distribution(self.remaining),
                "on_winning_turn_fraction":_rate(self.remaining[1],total),
                "within_two_turns_of_win_fraction":_rate(self.remaining[1]+self.remaining[2],total)}


def _audit_positions(root, db, scratch, metadata):
    import pyarrow as pa
    import pyarrow.parquet as pq

    expected_types = {
        "game_id":pa.string(),"board":pa.binary(24),"generation_seed":pa.int64(),
        **dict.fromkeys(("game_length","turn_number","position_id"),pa.uint32()),
        **dict.fromkeys(("white_agent","brown_agent","matchup_category"),pa.uint16()),
        **dict.fromkeys(("player_to_move","winner"),pa.int8()),
        **dict.fromkeys(("white_bar","brown_bar","win_target","game_phase","mixture_type","source_policy_metadata"),pa.uint8()),
    }
    scratch.execute("""CREATE TABLE exported(game_id TEXT,turn INTEGER,PRIMARY KEY(game_id,turn))
                    WITHOUT ROWID""")
    index = StateIndex(scratch)
    stats = {split:PositionStats() for split in SPLITS}
    for split in SPLITS:
        for file in sorted((root/split).rglob("*.parquet")):
            parquet = pq.ParquetFile(file)
            require(set(parquet.schema_arrow.names) == VALUE_COLUMNS, f"unexpected value columns in {file.name}")
            require(all(parquet.schema_arrow.field(k).type == v for k,v in expected_types.items()),
                    f"value column type differs in {file.name}")
            for batch in parquet.iter_batches(batch_size=2048):
                for row in batch.to_pylist():
                    gid, turn = row["game_id"], row["turn_number"]
                    stored = db.execute("""SELECT s.state,s.phase,s.policy,g.split,g.winner,g.turns,g.generation_seed,
                        g.white_agent,g.brown_agent,g.category,g.mixture_type FROM samples s JOIN games g USING(game_id)
                        WHERE s.game_id=? AND s.turn_number=?""", (gid,turn)).fetchone()
                    require(stored is not None, f"{gid}/{turn}: missing source sample")
                    board = row["board"]
                    require(isinstance(board,bytes) and len(board) == 24, f"{gid}/{turn}: wrong number of points")
                    actor, winner = row["player_to_move"], row["winner"]
                    require(actor in (-1,1) and winner in (-1,1), f"{gid}/{turn}: invalid actor or winner")
                    wire = board + struct.pack("BBb",row["white_bar"],row["brown_bar"],actor)
                    require((split, turn, row["game_length"], row["generation_seed"], wire, row["game_phase"], winner)
                            == (stored["split"], row["position_id"], stored["turns"], stored["generation_seed"],
                                stored["state"], stored["phase"], stored["winner"]), f"{gid}/{turn}: row metadata differs")
                    require(row["win_target"] in (0,1) and row["win_target"] == int(actor==winner),
                            f"{gid}/{turn}: wrong value target")
                    white = metadata["agents"][str(row["white_agent"])]
                    brown = metadata["agents"][str(row["brown_agent"])]
                    category = metadata["categories"][str(row["matchup_category"])]
                    require((white,brown,category) == (stored["white_agent"],stored["brown_agent"],stored["category"]),
                            f"{gid}/{turn}: policy identity or category differs")
                    require(metadata["mixture_types"].get(str(row["mixture_type"])) == stored["mixture_type"],
                            f"{gid}/{turn}: mixture type differs")
                    require(row["source_policy_metadata"] == stored["policy"], f"{gid}/{turn}: source policy differs")
                    strategy = metadata["strategies"][str(stored["policy"] & 127)]
                    try:
                        scratch.execute("INSERT INTO exported VALUES(?,?)", (gid,turn))
                    except sqlite3.IntegrityError as error:
                        raise AuditError(f"{gid}/{turn}: duplicate exported position") from error
                    index.add(wire,split)
                    stats[split].add(row,wire,white if actor==1 else brown,strategy,category,f"{white} vs {brown}")
    drops = Counter({split:0 for split in SPLITS})
    for gid,turn,state,split in db.execute(
        "SELECT s.game_id,s.turn_number,s.state,g.split FROM samples s JOIN games g USING(game_id)"
    ):
        if scratch.execute("SELECT 1 FROM exported WHERE game_id=? AND turn=?", (gid,turn)).fetchone():
            continue
        owner = scratch.execute("SELECT split FROM model_inputs WHERE state=?", (_actor_key(state),)).fetchone()
        require(owner and owner[0] != split, f"{gid}/{turn}: source row missing without cross-split duplicate")
        drops[split] += 1
    distribution_by_split = {}
    for split, pos in stats.items():
        histogram = Counter()
        for gid, in db.execute("SELECT game_id FROM games WHERE winner IS NOT NULL AND split=?", (split,)):
            count, = scratch.execute("SELECT COUNT(*) FROM exported WHERE game_id=?", (gid,)).fetchone()
            histogram[count] += 1
        distribution_by_split[split] = dict(pos.report(), samples_per_completed_game=distribution(histogram))
    return dict(index.report(), candidate_drops_by_split=dict(drops)), distribution_by_split, sum(drops.values())


def _audit_catalog(root, db, scratch):
    """Check the portable game catalog and packed trajectory slices against the evidence."""
    import pyarrow.parquet as pq

    scratch.execute("CREATE TABLE catalog(game_id TEXT PRIMARY KEY) WITHOUT ROWID")
    retained = Counter()
    count = 0
    for path in sorted((root/"games").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                gid = row["game_id"]
                stored = db.execute("SELECT * FROM games WHERE game_id=?", (gid,)).fetchone()
                require(stored is not None, f"{gid}: unknown portable game")
                require(scratch.execute("INSERT OR IGNORE INTO catalog VALUES(?)",(gid,)).rowcount == 1,
                        f"{gid}: repeated portable game")
                relative = row.pop("trajectory_file")
                offset = row.pop("trajectory_offset")
                length = row.pop("trajectory_size")
                require(set(row) == set(stored.keys()) - {"trajectory"} and all(
                    row[key] == stored[key] for key in row), f"{gid}: portable game metadata differs")
                if stored["trajectory"] is None:
                    require(relative is None and offset is None and length is None, f"{gid}: unexpected trajectory")
                else:
                    require(relative and offset is not None and offset>=0 and length==len(stored["trajectory"]),
                            f"{gid}: bad trajectory slice")
                    full = (root/relative).resolve()
                    require(full.parent == (root/"trajectories").resolve(), f"{gid}: invalid trajectory location")
                    with full.open("rb") as handle:
                        handle.seek(offset)
                        require(handle.read(length) == stored["trajectory"], f"{gid}: portable trajectory differs")
                    retained[row["category"]] += 1
                count += 1
    require(count == db.execute("SELECT COUNT(*) FROM games").fetchone()[0], "incomplete portable game catalog")
    return dict(retained)


def _display(value, fmt=",.1f"):
    return "—" if value is None else format(value,fmt)


def _markdown(r):
    primary = r["games"]["primary"]
    reach = r["validation"]
    d = r["diversity"]
    out = ["# Synthetic dataset audit", "",
           f"The archive contains {r['attempted_games']:,} attempts and {r['completed_games']:,} completed "
           f"games. The value partitions contain {d['positions']:,} pre-roll positions and "
           f"{d['unique']:,} distinct boards including active color. The primary allocation reached "
           f"{r['allocation']['primary_completed']:,} of {r['allocation']['primary_target']:,} completions. "
           f"The separate generalization set reached {r['allocation']['heldout_completed']:,} of "
           f"{r['allocation']['heldout_target']:,}.", "",
           "## Validation", "",
           f"The audit checked checkpoint labels, deterministic allocations, player-to-move parity, phase assignments, "
           f"policy traces and the exported Parquet rows. It replayed {reach.get('trajectory_games',0):,} archived "
           f"games and independently checked {reach.get('trajectory_checked_samples',0):,} sampled states against "
           f"their exact turn. It regenerated {reach.get('regenerated_games',0):,} additional games from their seeds. "
           f"{reach.get('games_without_offline_replay',0):,} games ({reach.get('samples_without_offline_replay',0):,} "
           f"sampled states) rely on the simulator's online action and reachability checks.", "",
           f"Exact repeated positions within splits account for {_display(d['repeat_fraction'],'.2%')} of rows. "
           f"{r['cross_split_drops']:,} candidate rows were withheld to prevent equal board/turn-color inputs "
           f"from appearing in different splits after player normalization. The exported partitions have "
           f"{d['unique_model_inputs']:,} distinct player-normalized model inputs and no cross-split overlap.", "",
           "## Sampling and labels", "",
           "| Split | Positions | Win labels | Early | Middle | Late | Near terminal | On winning turn |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, s in r["positions"].items():
        p = s["phases"]
        out.append(f"| {split} | {s['count']:,} | {_display(s['winning_label_fraction'],'.1%')} | "
                   + " | ".join(_display(_rate(p.get(phase,0), s["count"]), ".1%") for phase in PHASES)
                   + f" | {_display(s['on_winning_turn_fraction'],'.1%')} |")
    out.extend(["", "Phase targets (early, middle, late, near terminal) are "
                + ", ".join(_display(w,".0%") for w in r["target_phase_weights"]) + ". "
                "The final column counts sampled states immediately before the winning turn. "
                "The near-terminal phase uses board geometry.", "",
                "## Completion and game lengths", "",
                "| Cohort | Attempts | Wins | Unfinished | Completion | Median completed turns | 90th percentile | Opening player win rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for cohort, c in r["games"].items():
        o = c["first_player"]
        out.append(f"| {cohort} | {o['attempted']:,} | {o['completed']:,} | {o['unfinished']:,} | "
                   f"{_display(o['completion_rate'],'.1%')} | {_display(c['completed_game_turns']['median'])} | "
                   f"{_display(c['completed_game_turns']['p90'])} | {_display(o['win_rate_completed'],'.1%')} |")
    out.extend(["", "Opening-player rates use completed games. Each observed length includes the final turn "
                "or, for unfinished attempts, every turn through the limit.", "",
                "### Primary allocation categories", "",
                "| Category | Target | Attempts | Completed | Unfinished | Completion rate | Mean observed turns |",
                "|---|---:|---:|---:|---:|---:|---:|"])
    for name, row in primary["by_category"].items():
        out.append(f"| {name} | {r['allocation']['by_category'].get(name,0):,} | {row['attempted']:,} | {row['completed']:,} | {row['unfinished']:,} | "
                   f"{_display(row['completion_rate'],'.1%')} | {_display(row['mean_observed_turns'])} |")
    out.extend(["", "### Primary agent seats", "",
                "| Policy | Appearances | Completed | Win rate on completions | Mean observed turns |",
                "|---|---:|---:|---:|---:|"])
    for name, row in primary["by_agent"].items():
        out.append(f"| {name} | {row['attempted']:,} | {row['completed']:,} | "
                   f"{_display(row['win_rate_completed'],'.1%')} | {_display(row['mean_observed_turns'])} |")
    out.extend(["", "A self-play game contributes two agent appearances. Comparisons reflect the assigned "
                "opponents. Counts for each White/Brown matchup are available in the JSON report.", "",
                "## Primary game behavior", "",
                "| Metric | Observed value |", "|---|---:|"])
    for key, value in primary["behavior"].items():
        out.append(f"| {key.replace('_',' ')} | {_display(value,',.3f')} |")
    out.extend(["", "Per-attempt totals include unfinished games. Blots, bar, concentration and blockade "
                "use the active player's position after its turn. Eligibility uses both players after each turn.", "",
                "## Performance", "", "| Measurement | Value |", "|---|---:|"])
    for key, value in r["performance"].items():
        if isinstance(value,(float,int)) or value is None:
            out.append(f"| {key.replace('_',' ')} | {_display(value,',.3f')} |")
    out.extend(["", "Generation wall time and worker time measure different quantities. Mean worker simulation time "
                "is per attempted game. Serialization speed uses bytes written divided by measured serialization time. "
                "A dash means a measurement was unavailable. " + r["performance"].get("memory_scope",""),
                "", "## Observations", ""])
    out.extend("- " + line for line in r["observations"])
    return "\n".join(out) + "\n"


def _observations(cohorts, positions, phase_targets, diversity):
    primary = cohorts["primary"]
    remarks = []
    eligible = [m for m in primary["by_matchup"] if m["attempted"] >= 5]
    if eligible:
        most = min(eligible, key=lambda x:(x["completion_rate"], -x["attempted"]))
        remarks.append(f"Among color-specific matchups with at least five attempts, {most['white']} (White) vs "
                       f"{most['brown']} (Brown) had the lowest completion rate at {most['completed']}/{most['attempted']}.")
    seats = primary["by_agent"]
    if seats:
        slow = max(seats,key=lambda k:seats[k]["mean_observed_turns"])
        remarks.append(f"{slow} had the longest mean observed game length among primary policy seats "
                       f"({_display(seats[slow]['mean_observed_turns'])} turns across {seats[slow]['attempted']:,} appearances).")
    exploratory = [(s,v) for s,v in seats.items() if (s.startswith("noisy:") or s=="random") and v["completed"]>=5]
    if exploratory:
        policy, value = min(exploratory,key=lambda pair:(pair[1]["win_rate_completed"],-pair[1]["completed"]))
        remarks.append(f"Among exploratory seats with at least five completed appearances, {policy} had the "
                       f"lowest observed win rate ({value['wins']}/{value['completed']}). Opponents and sample "
                       "sizes differ across policies.")
    counts = Counter()
    labels = Counter()
    for split in ("train","validation","test"):
        counts.update(positions[split]["phases"])
        labels.update(positions[split]["labels"])
    if counts:
        total = sum(counts.values())
        gaps = [(target - counts[p]/total,p,target) for p,target in zip(PHASES,phase_targets)]
        gap, phase, target = max(gaps)
        if gap > .001:
            remarks.append(f"The largest primary phase shortfall was {phase}. Its sampled share was "
                           f"{counts[phase]/total:.1%} against a {target:.1%} target.")
        remarks.append(f"From the side to move, {labels[1]/total:.1%} of primary positions have a winning label.")
    if diversity["positions"]:
        most = max(s["max_visits"] for s in diversity["by_split"].values())
        remarks.append(f"The most frequently sampled exact board and active color appeared {most:,} times.")
    return remarks or ["No completed sample is available for a diversity comparison."]


def _checksums(root, manifest):
    """Check the manifest's content hashes in fixed-size chunks."""
    entries = manifest["checksums"]
    portable_files = {"config.json","agent_metadata.json"}
    for folder in (*SPLITS,"games","trajectories"):
        portable_files.update(str(p.relative_to(root)) for p in (root/folder).rglob("part-*") if p.is_file())
    require(portable_files.issubset(entries),"portable files missing from checksum manifest")
    for relative, expected in entries.items():
        path = root / relative
        require(path.is_file(), f"missing file {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for part in iter(lambda: file.read(1024*1024), b""):
                digest.update(part)
        expected = expected.get("sha256") if isinstance(expected,dict) else expected
        require(digest.hexdigest() == expected, f"checksum differs for {relative}")
    require(sum((root/p).stat().st_size for p in entries) == manifest["performance"]["storage_bytes"],
            "portable storage byte count differs")
    return len(entries)


def audit(dataset_dir, *, verify_seeds=0, write_report=True):
    """Check exported partitions, replay retained games, and optionally regenerate other attempts.

    verify_seeds=-1 regenerates every attempt without a stored trajectory; a positive
    limit regenerates that many in deterministic slot/attempt order. Failures raise AuditError.
    """
    from synthetic_game import SamplingConfig
    from synthetic_plan import PlanConfig, make_plan, spec_for
    from synthetic_store import evidence_db, write_json
    from agents import AGENT_NAMES

    require(isinstance(verify_seeds,int) and verify_seeds >= -1, "invalid seed verification count")
    start = perf_counter()
    root = Path(dataset_dir)
    manifest = json.loads((root/"manifest.json").read_text())
    config = json.loads((root/"config.json").read_text())
    metadata = json.loads((root/"agent_metadata.json").read_text())
    require(manifest["config"] == config,"manifest configuration differs from run configuration")
    require(metadata["phases"] == {str(i):p for i,p in enumerate(PHASES)}
            and metadata["strategies"] == {str(i):n for i,n in enumerate(AGENT_NAMES)},
            "phase or policy dictionary is inconsistent with recorded data")
    require(metadata["policy_config_version"] == config["software"]["decision_sha256"]
            and metadata["mixture_weights"] == config["plan"]["mixture_weights"]
            and metadata["heldout_weights"] == config["plan"]["heldout_weights"]
            and metadata["epsilons"] == config["plan"]["epsilons"] and metadata["exploration_flag"] == 128,
            "strategy metadata differs from the recorded configuration")
    checked_files = _checksums(root,manifest)
    plan = PlanConfig(**config["plan"])
    slots = {str(slot.slot_id):slot for slot in make_plan(plan)}
    sampling = SamplingConfig(**config["sampling"])

    def plans(entry):
        return spec_for(slots[str(entry["slot_id"])],entry["attempt"],plan),plan

    with evidence_db(root) as db, closing(sqlite3.connect("")) as scratch:
        db.row_factory = sqlite3.Row
        require(db.execute("PRAGMA quick_check").fetchone()[0] == "ok","source database integrity check failed")
        require(db.execute("""SELECT 1 FROM games GROUP BY generation_seed HAVING COUNT(*)>1 LIMIT 1""").fetchone()
                is None, "multiple attempts share one generation seed")
        require(db.execute("""SELECT 1 FROM games GROUP BY slot_id HAVING SUM(winner IS NOT NULL)>1 LIMIT 1""").fetchone()
                is None, "more than one completed game for an allocation slot")
        games,coverage = _read_games(db,sampling.phase_thresholds,plans,sampling,verify_seeds)
        coverage["retained_trajectories_by_category"] = _audit_catalog(root,db,scratch)
        diversity,positions,dropped = _audit_positions(root,db,scratch,metadata)

    observed = sum(c["first_player"]["attempted"] for c in games.values())
    completed = sum(c["first_player"]["completed"] for c in games.values())
    require((observed,completed,observed-completed,diversity["positions"]) ==
            tuple(manifest[k] for k in ("attempted_games","completed_games","unfinished_games","sampled_positions")),
            "manifest totals differ")
    require({k:v["count"] for k,v in positions.items()} == manifest["split_sizes"], "manifest split position counts differ")
    allocation = {"primary_target":plan.target_games, "heldout_target":plan.heldout_games,
                  "primary_completed":games["primary"]["first_player"]["completed"],
                  "heldout_completed":games["heldout_evaluation"]["first_player"]["completed"],
                  "by_category":dict(Counter(s.category for s in slots.values() if s.split!="heldout_evaluation"))}
    require((allocation["by_category"],allocation["heldout_target"],allocation["primary_completed"],allocation["heldout_completed"])
            == (manifest["allocation_targets"],manifest["heldout_target"],manifest["primary_completed"],manifest["heldout_completed"]),
            "manifest allocation differs")
    for split in SPLITS:
        cohort = games["heldout_evaluation" if split=="heldout_evaluation" else "primary"]
        stats = cohort["by_split"].get(split,{"attempted":0,"completed":0})
        require(all(manifest["split_games"][split][k] == stats[k] for k in ("attempted","completed")),
                "manifest split game counts differ")
    # Portable archives reconstruct evidence from released rows. They retain paired game
    # trajectories but may omit candidates removed before Parquet export.
    has_candidate_checkpoint = (root/"checkpoints"/"run.sqlite3").is_file()
    cross_split = manifest["cross_split_drops"]
    num_drops = sum(cross_split.values()) if isinstance(cross_split,dict) else cross_split
    if has_candidate_checkpoint:
        require(dropped == num_drops, "cross-split removal count differs")
        if isinstance(cross_split,dict):
            require(cross_split == diversity["candidate_drops_by_split"], "cross-split removal counts per split differ")
        require(coverage["candidate_positions_examined"] == manifest["binary_rows_before_filter"],
                "candidate sample count differs")
    performance = dict(manifest["performance"])
    performance["mean_worker_simulation_seconds_per_attempt"] = _rate(performance["simulation_worker_seconds"],observed)
    performance["serialization_bytes_per_second"] = _rate(performance["serialization_bytes"],
                                                            performance["serialization_seconds"])
    if not has_candidate_checkpoint:
        diversity["candidate_drops_by_split"] = None
    coverage["checked_file_sha256"] = checked_files
    coverage["has_prefilter_checkpoint"] = has_candidate_checkpoint
    coverage["audit_seconds"] = perf_counter() - start
    result = {"status":"passed", "attempted_games":observed,"completed_games":completed,
              "unfinished_games":observed-completed,"games":games,"positions":positions,
              "target_phase_weights":sampling.phase_weights,"diversity":diversity,"validation":coverage,
              "cross_split_drops":num_drops,"performance":performance,"allocation":allocation,
              "observations":_observations(games,positions,sampling.phase_weights,diversity)}
    if write_report:
        destination = root/"reports"
        destination.mkdir(parents=True,exist_ok=True)
        write_json(destination/"report.json",result)
        pending = destination/"REPORT.md.tmp"
        pending.write_text(_markdown(result))
        pending.replace(destination/"REPORT.md")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--verify-seeds",type=int,default=0,help="regenerate N additional unarchived games; -1 for all")
    args = parser.parse_args()
    report = audit(args.dataset,verify_seeds=args.verify_seeds)
    print(f"Validated {report['attempted_games']:,} attempts and {report['diversity']['positions']:,} pre-roll positions.")
