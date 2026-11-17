"""Seeded complete-game simulation and paired-color tournaments. See SIMULATION.md."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations, combinations_with_replacement, islice
import json
import math
from pathlib import Path
from random import Random
from typing import Optional

import backgammon as game
from agents import AGENT_NAMES, create_agent
from features import evaluate_chosen, feature_record
from selfplay import DEFAULT_MAX_TURNS
from training_data import PositionLog, pack_sample, ranked_decision_inputs


DEFAULT_SEED = 2026
ACTIONS = (
    "pass", "only_forward", "only_backward", "mixed", "capture", "self_capture",
    "bar_entry", "bear_off", "stack_join", "reduce_vulnerability", "extend_blockade",
    "increase_concentration", "increase_threats", "advance_home_half", "disrupt_eligibility",
)


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _agent(spec):
    return create_agent(spec) if isinstance(spec, str) else spec


@dataclass
class GameResult:
    game_id: str
    seed: int
    white: str
    brown: str
    winner: Optional[int]
    turns: int
    white_stats: Counter
    brown_stats: Counter
    samples: list = field(default_factory=list, repr=False)


def _observe(stats, other, candidate):
    before, after = candidate.own_before, candidate.own_after
    back = candidate.backward_moves
    forward = len(candidate.turn.moves) - back
    stats.update({
        "own_turns": 1, "checker_moves": forward + back, "forward_steps": forward, "backward_steps": back,
        "captures": candidate.captures, "self_captures": candidate.self_captures,
        "forward_pips": candidate.forward_progress, "backward_pips": candidate.backward_displacement,
        "blots": after.blots, "bar": after.bar, "concentration": after.stack_concentration,
        "occupied": after.occupied_points, "made": after.made_points, "blockade": after.longest_blockade,
        "vulnerability": after.effective_vulnerability, "victory_distance": after.victory_distance,
        "victory_gain": before.victory_distance - after.victory_distance,
        "eligible_plies": int(after.eligible), "board_plies": 1,
        "pass": int(not forward and not back), "only_forward": int(forward > 0 and not back),
        "only_backward": int(back > 0 and not forward), "mixed": int(bool(back and forward)),
        "capture": int(candidate.captures > 0), "self_capture": int(candidate.self_captures > 0),
        "bar_entry": int(any(m.source == game.BAR for m in candidate.turn.moves)), "bear_off": int(candidate.wins),
        "stack_join": int(bool(candidate.stack_joins)), "reduce_vulnerability": int(after.effective_vulnerability < before.effective_vulnerability),
        "extend_blockade": int(after.longest_blockade > before.longest_blockade),
        "increase_concentration": int(after.stack_concentration > before.stack_concentration),
        "increase_threats": int(after.threats > before.threats),
        "advance_home_half": int(after.distance_to_home_half < before.distance_to_home_half),
        "disrupt_eligibility": int(candidate.opp_before.eligible and not candidate.opp_after.eligible),
    })
    other["board_plies"] += 1
    other["eligible_plies"] += int(candidate.opp_after.eligible)


def _play(white, brown, seed, max_turns, game_id, sample_every, *, observer=None):
    # Independent die, per-agent action, and logging streams keep dice pairing
    # intact across passes, different game lengths, and random decision counts.
    dice_rng = Random(seed)
    action_rng = {
        color: Random(f"action/{seed}/{agent.name}/{color if white.name == brown.name else 0}")
        for color, agent in ((game.WHITE, white), (game.BROWN, brown))
    }
    sample_rng = Random(f"samples/{seed}") if sample_every else None
    agents = {game.WHITE: white, game.BROWN: brown}
    stats = {game.WHITE: Counter(), game.BROWN: Counter()}
    actor, roll = game.opening_roll(dice_rng)
    state = game.starting_state(actor)
    if observer is not None:
        observer.opened(state)
    samples = []
    for number in range(1, max_turns + 1):
        actor = int(state.player)
        agent = agents[actor]
        choose_candidate = getattr(agent, "choose_candidate", None)
        if choose_candidate is None:
            chosen = agent.choose_turn(state, roll, rng=action_rng[actor])
            bar_penalty = getattr(getattr(agent, "config", None), "bar_penalty", 3)
            features = evaluate_chosen(state, chosen, bar_penalty=bar_penalty)
        else:
            features = choose_candidate(state, roll, rng=action_rng[actor])
            chosen = features.turn
        if observer is not None:
            observer.chosen(number, state, roll, agent, features)
        _observe(stats[actor], stats[-actor], features)
        if sample_rng is not None and (sample_every == 1 or sample_rng.randrange(sample_every) == 0):
            samples.append(pack_sample(number, state, roll, chosen, feature_record(features)))
        state = chosen.state
        winner = game.winner(state)
        if winner is not None or number == max_turns:
            return GameResult(game_id, seed, white.name, brown.name, winner, number,
                              stats[game.WHITE], stats[game.BROWN], samples)
        if observer is not None:
            observer.before_roll(number + 1, state)
        roll = game.roll_dice(dice_rng)


def play_game(agent_white, agent_brown, seed=None, *, max_turns=DEFAULT_MAX_TURNS, game_id=None,
              positions=None, sample_every=1):
    """Play one complete or capped game. Accept registry names or configured agents."""
    white, brown = _agent(agent_white), _agent(agent_brown)
    seed = DEFAULT_SEED if seed is None else game._integer(seed, "seed")
    game_id = game_id or f"{white.name}-vs-{brown.name}/{seed}"
    result = _play(white, brown, seed, _positive(max_turns, "max_turns"), game_id,
                   _positive(sample_every, "sample_every") if positions is not None else 0)
    if positions is not None:
        with PositionLog(positions) as log:
            log.write(result)
    return result


def _ratio(a, b):
    return a / b if b else None


def _duration(hist):
    n = sum(hist.values())
    if not n:
        return {"count": 0, "mean": None, "median": None, "histogram": {}}
    halfway = ((n - 1) // 2, n // 2)
    middle = []
    count = 0
    for length in sorted(hist):
        middle.extend(length for ix in halfway if count <= ix < count + hist[length])
        count += hist[length]
    return {"count": n, "mean": sum(k * v for k, v in hist.items()) / n,
            "median": sum(middle) / 2, "histogram": dict(sorted(hist.items()))}


def _outcomes(wins, losses, unfinished):
    completed = wins + losses
    starts = completed + unfinished
    return {"appearances": starts, "wins": wins, "losses": losses, "unfinished": unfinished,
            "win_fraction_all": _ratio(wins, starts), "win_fraction_completed": _ratio(wins, completed)}


def _behavior(stats, appearances):
    t = stats["own_turns"]
    return {
        "actor_turns": t, "checker_moves": stats["checker_moves"],
        "captures_per_game": _ratio(stats["captures"], appearances),
        "self_captures_per_game": _ratio(stats["self_captures"], appearances),
        "captures_per_turn": _ratio(stats["captures"], t),
        "capture_turn_fraction": _ratio(stats["capture"], t),
        "forward_step_fraction": _ratio(stats["forward_steps"], stats["checker_moves"]),
        "backward_step_fraction": _ratio(stats["backward_steps"], stats["checker_moves"]),
        "forward_pips_per_turn": _ratio(stats["forward_pips"], t),
        "backward_pips_per_turn": _ratio(stats["backward_pips"], t),
        "forward_pips_per_game": _ratio(stats["forward_pips"], appearances),
        "backward_pips_per_game": _ratio(stats["backward_pips"], appearances),
        "mean_blots": _ratio(stats["blots"], t), "mean_bar": _ratio(stats["bar"], t),
        "mean_vulnerability": _ratio(stats["vulnerability"], t),
        "mean_concentration": _ratio(stats["concentration"], t),
        "mean_occupied": _ratio(stats["occupied"], t), "mean_made": _ratio(stats["made"], t),
        "mean_blockade": _ratio(stats["blockade"], t),
        "mean_victory_distance": _ratio(stats["victory_distance"], t),
        "mean_victory_gain": _ratio(stats["victory_gain"], t),
        "eligible_fraction": _ratio(stats["eligible_plies"], stats["board_plies"]),
        "eligible_plies_per_game": _ratio(stats["eligible_plies"], appearances),
        "action_counts": {k: stats[k] for k in ACTIONS},
        "action_turn_fractions": {k: _ratio(stats[k], t) for k in ACTIONS},
    }


@dataclass
class Aggregate:
    observed: Counter = field(default_factory=Counter)
    finished: Counter = field(default_factory=Counter)
    unfinished: Counter = field(default_factory=Counter)
    winners: Counter = field(default_factory=Counter)
    stats: Counter = field(default_factory=Counter)

    def add(self, length, winner, stats):
        self.observed[length] += 1
        self.winners[winner] += 1
        (self.unfinished if winner is None else self.finished)[length] += 1
        self.stats.update(stats)

    def duration(self):
        return {"all_observed": _duration(self.observed), "completed": _duration(self.finished),
                "unfinished_observed": _duration(self.unfinished)}


PROFILE_METRICS = ("capture_turn_fraction", "backward_step_fraction", "mean_blots", "mean_bar",
                   "mean_concentration", "mean_occupied", "mean_blockade", "mean_victory_gain",
                   "eligible_fraction", "average_game_length")
FOCUS_PAIRS = (("fortress", "hoarder"), ("baseline", "assassin"), ("fortress", "blockader"),
               ("sprinter", "optimizer"), ("saboteur", "assassin"))


def compare_profiles(profiles, threshold=0.12):
    """Mean absolute gap in cohort-range units, ignoring constant/unavailable metrics."""
    ranges = {}
    for key in PROFILE_METRICS:
        values = [p[key] for p in profiles.values() if p[key] is not None]
        if values and max(values) > min(values):
            ranges[key] = max(values) - min(values)
    pairs = []
    for a, b in combinations(profiles, 2):
        gaps = {k: abs(profiles[a][k] - profiles[b][k]) / span for k, span in ranges.items()
                if profiles[a][k] is not None and profiles[b][k] is not None}
        # Exact ties across all defined constant features are identical on this workload.
        observed = profiles[a].get("actor_turns", 1) > 0 and profiles[b].get("actor_turns", 1) > 0
        shared = any(profiles[a][k] is not None and profiles[b][k] is not None for k in PROFILE_METRICS)
        distance = None
        if observed and shared:
            distance = sum(gaps.values()) / len(gaps) if gaps else 0.0
        differences = sorted(gaps, key=lambda k: (-gaps[k], k))[:3]
        pairs.append({"agents": [a, b], "distance": distance,
                      "near": distance is not None and distance <= threshold,
                      "largest_gaps": [{"metric": k, "a": profiles[a][k], "b": profiles[b][k],
                                       "normalized_gap": gaps[k]} for k in differences]})
    pairs.sort(key=lambda p: (float("inf") if p["distance"] is None else p["distance"], p["agents"]))
    lookup = {frozenset(p["agents"]): p for p in pairs}
    return {"threshold": threshold, "metrics": list(PROFILE_METRICS), "observed_ranges": ranges,
            "pairs_by_similarity": pairs, "flagged_near_pairs": [p["agents"] for p in pairs if p["near"]],
            "requested_comparisons": [lookup[frozenset(pair)] for pair in FOCUS_PAIRS if frozenset(pair) in lookup]}


def decision_agreement(positions, limit=100, *, seed=DEFAULT_SEED):
    """Compare deterministic choices on non-forced rolls stratified by source actor."""
    limit = _positive(limit, "limit")
    seed = game._integer(seed, "seed")
    names = dict.fromkeys(name for pair in FOCUS_PAIRS for name in pair)
    agents = {name: create_agent(name) for name in names}
    counts = {pair: Counter() for pair in FOCUS_PAIRS}
    source_agents = Counter()
    fingerprint = sha256()
    selected = forced = 0
    with ranked_decision_inputs(positions, seed) as (rows, sizes, groups):
        order = sorted(groups)
        Random(seed).shuffle(order)
        rotation = deque(order)
        while rotation and selected < limit:
            actor = rotation.popleft()
            for (key,) in groups[actor]:
                state = game.deserialize_state(key[:game.STATE_SIZE])
                roll = tuple(key[game.STATE_SIZE:])
                if len(game.legal_turns(state, roll)) <= 1:
                    forced += 1
                    continue
                chosen = {name: agent.choose_turn(state, roll) for name, agent in agents.items()}
                for pair, count in counts.items():
                    first, second = (chosen[name] for name in pair)
                    count["exact"] += int(first.moves == second.moves)
                    count["result"] += int(first.state == second.state)
                selected += 1
                source_agents[actor] += 1
                fingerprint.update(key)
                rotation.append(actor)
                break
    return {
        "source": Path(positions).name, "selection_seed": seed, "requested": limit, "positions_compared": selected,
        "logged_rows_examined": rows, "repeated_position_roll_skipped": rows - sum(sizes.values()),
        "unique_inputs_by_source": sizes, "inputs_checked_for_alternatives": forced + selected,
        "forced_position_roll_skipped": forced, "source_actors": {actor: source_agents[actor] for actor in sorted(sizes)},
        "selection_sha256": fingerprint.hexdigest(),
        "selection": ("Deduplicate state and unordered roll by earliest archive occurrence. Rotate across source actors "
                      "in seed-shuffled order, taking each actor's next seeded hash-ranked input with at least two "
                      "legal final boards. Redistribute unfilled slots to the remaining actors."),
        "pairs": [
            {"agents": list(pair), "same_moves": count["exact"], "same_final_board": count["result"],
             "same_moves_fraction": _ratio(count["exact"], selected),
             "same_final_board_fraction": _ratio(count["result"], selected)}
            for pair, count in counts.items()
        ],
    }


def _execute(task):
    a, b, index, seed, limit, sampling = task
    a_white = index % 2 == 0
    left, right = _agent(a), _agent(b)
    white, brown = (left, right) if a_white else (right, left)
    identifier = f"{left.name}-vs-{right.name}/{index}/{seed}"
    result = _play(white, brown, seed, limit, identifier, sampling)
    return (left.name, right.name), a_white, result


def _results(schedule, workers):
    if workers == 1:
        yield from map(_execute, schedule)
        return
    # Bound work in flight, including any sampled trajectories. Preserve job order
    # for reproducible floating-point aggregation and deterministic log ordering.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = deque(pool.submit(_execute, task) for task in islice(schedule, workers * 2))
        while pending:
            result = pending.popleft().result()
            task = next(schedule, None)
            if task is not None:
                pending.append(pool.submit(_execute, task))
            yield result


def _run(pairs, n_games, seed, max_turns, workers, positions, sample_every, near_threshold):
    n_games = _positive(n_games, "n_games")
    workers = _positive(workers, "workers")
    max_turns = _positive(max_turns, "max_turns")
    sampling = _positive(sample_every, "sample_every") if positions is not None else 0
    seed = game._integer(seed, "seed")
    if not math.isfinite(near_threshold) or not 0 <= near_threshold <= 1:
        raise ValueError("near_threshold must be between 0 and 1")
    pairs = list(pairs)
    aggregates, agent_stats, color_wins = {}, {}, Counter()
    planned_pairs = []
    for a, b in pairs:
        an, bn = _agent(a).name, _agent(b).name
        aggregates[(an, bn)] = Aggregate()
        agent_stats.setdefault(an, Aggregate())
        agent_stats.setdefault(bn, Aggregate())
        planned_pairs.append((a, b, an == bn))
    # Two different policies play each die stream in both colors.
    schedule = ((a,b,i,seed + (i if same else i // 2),max_turns,sampling)
                for a,b,same in planned_pairs for i in range(n_games))

    total = Aggregate()
    with (PositionLog(positions) if positions is not None else nullcontext()) as log:
        for pair, a_white, r in _results(schedule, workers):
            if log is not None:
                log.write(r)
            winner_seat = None if r.winner is None else ("A" if (r.winner == game.WHITE) == a_white else "B")
            combined = r.white_stats.copy()
            combined.update(r.brown_stats)
            aggregates[pair].add(r.turns, winner_seat, combined)
            total.add(r.turns, r.winner, combined)
            color_wins[(pair, r.winner)] += 1
            for color, name, stats in ((game.WHITE, r.white, r.white_stats), (game.BROWN, r.brown, r.brown_stats)):
                role = None if r.winner is None else ("W" if color == r.winner else "L")
                agent_stats[name].add(r.turns, role, stats)

    matchups = []
    matrix = {name: {} for name in agent_stats}
    for pair, agg in aggregates.items():
        a, b = pair
        w, l, u = (agg.winners[k] for k in ("A","B",None))
        aw, bw = _outcomes(w,l,u), _outcomes(l,w,u)
        n = w+l+u
        matchups.append({"agents": list(pair), "games": n, "a": aw, "b": bw, "completed": w+l,
                         "unfinished": u, "white_wins": color_wins[pair,game.WHITE],
                         "brown_wins": color_wins[pair,game.BROWN], "length": agg.duration(),
                         "behavior": _behavior(agg.stats,n)})
        if a == b:
            matrix[a][a] = {"games": n, **_outcomes(w+l,w+l,2*u)}
        else:
            matrix[a][b] = {"games": n, **aw}
            matrix[b][a] = {"games": n, **bw}
    profiles = {}
    for name, agg in agent_stats.items():
        results = _outcomes(agg.winners["W"],agg.winners["L"],agg.winners[None])
        durations = agg.duration()
        profiles[name] = {**results, **_behavior(agg.stats, results["appearances"]),
                          "average_game_length": durations["all_observed"]["mean"], "length": durations}
    count = sum(total.observed.values())
    return {
        "seed": seed, "max_turns": max_turns, "games_per_pairing": n_games,
        "pairings": len(pairs), "games": count, "completed": sum(total.finished.values()),
        "unfinished": total.winners[None], "white_wins": total.winners[game.WHITE],
        "brown_wins": total.winners[game.BROWN], "length": total.duration(),
        "behavior": _behavior(total.stats,count), "matchups": matchups, "matrix": matrix,
        "profiles": profiles, "diversity": compare_profiles(profiles,near_threshold),
    }


def run_matchup(agent_a, agent_b, n_games, *, seed=DEFAULT_SEED, max_turns=DEFAULT_MAX_TURNS,
               workers=1, positions=None, sample_every=1, near_threshold=0.12):
    """Alternate seats, reusing each seed for a two-game color swap except on self-play."""
    return _run([(agent_a, agent_b)], n_games, seed, max_turns, workers, positions, sample_every, near_threshold)


def run_round_robin(agents, n_games_per_pairing, *, seed=DEFAULT_SEED, max_turns=DEFAULT_MAX_TURNS,
                    workers=1, positions=None, sample_every=1, near_threshold=0.12):
    """All unordered pairings including self-play; names must be distinct."""
    agents = tuple(agents)
    names = tuple(_agent(a).name for a in agents)
    if not agents or len(set(names)) != len(names):
        raise ValueError("supply one or more agents with distinct names")
    return _run(combinations_with_replacement(agents, 2), n_games_per_pairing, seed,
                max_turns, workers, positions, sample_every, near_threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("matchup")
    single.add_argument("agent_a", choices=AGENT_NAMES)
    single.add_argument("agent_b", choices=AGENT_NAMES)
    tournament = commands.add_parser("tournament")
    tournament.add_argument("--agents", nargs="+", choices=AGENT_NAMES, default=AGENT_NAMES)
    for command in (single, tournament):
        command.add_argument("--games", type=int, default=4, help="starts per unordered pairing")
        command.add_argument("--seed", type=int, default=DEFAULT_SEED)
        command.add_argument("--max-turns", "--turn-limit", type=int, default=DEFAULT_MAX_TURNS)
        command.add_argument("--workers", type=int, default=1)
        command.add_argument("--positions", type=Path, help="append sampled positions to a SQLite file")
        command.add_argument("--sample-every", type=int, default=1, help="sample with probability 1/N")
        command.add_argument("--near-threshold", type=float, default=0.12)
        command.add_argument("--out", type=Path, help="write the JSON report here; default is standard output")
    agreement = commands.add_parser("agreement", help="compare actual choices on stored positions")
    agreement.add_argument("positions", type=Path)
    agreement.add_argument("--limit", type=int, default=100)
    agreement.add_argument("--seed", type=int, default=DEFAULT_SEED, help="position-selection seed")
    agreement.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "agreement":
        report = decision_agreement(args.positions, args.limit, seed=args.seed)
        summary = f"{report['positions_compared']} positions compared."
    else:
        kwargs = {k: getattr(args, k) for k in ("seed","max_turns","workers","positions","sample_every","near_threshold")}
        if args.command == "matchup":
            report = run_matchup(args.agent_a, args.agent_b, args.games, **kwargs)
        else:
            report = run_round_robin(args.agents, args.games, **kwargs)
        summary = f"{report['games']} games, {report['completed']} wins, {report['unfinished']} unfinished."
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{summary} {args.out}")
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
