"""Time the actual feature/decision pipeline and memory use during streamed simulations."""

import argparse
from itertools import combinations_with_replacement, cycle, islice
import json
from pathlib import Path
import platform
from statistics import mean, median
import sys
from time import perf_counter

import backgammon as game
from agents import AGENT_NAMES, create_agent
from benchmark import cases
from features import iter_candidates
from heuristic import _legal_paths
from simulation import play_game
from selfplay import DEFAULT_MAX_TURNS


def peak_rss_mib():
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 2 if sys.platform == "darwin" else 1024)


def simulation_sample(games, limit, seed):
    """Measure production simulation including statistics, in one worker without logging."""
    names = list(combinations_with_replacement(AGENT_NAMES, 2))
    lengths, finished = [], []
    base_rss = peak_rss_mib()
    started = perf_counter()
    for i, (white, brown) in enumerate(islice(cycle(names), games)):
        if i // len(names) % 2:
            white, brown = brown, white
        result = play_game(white, brown, seed=seed + i, max_turns=limit)
        lengths.append(result.turns)
        if result.winner is not None:
            finished.append(result.turns)
    seconds = perf_counter() - started
    return dict(starts=games, finished=len(finished), unfinished=games-len(finished), seconds=seconds,
                turns=sum(lengths), starts_per_second=games/seconds, finishes_per_second=len(finished)/seconds,
                decisions_per_second=sum(lengths)/seconds, mean_observed_turns=mean(lengths),
                mean_finished_turns=mean(finished) if finished else None, wall_seconds_per_start=seconds/games,
                peak_rss_before_mib=base_rss, peak_rss_after_mib=peak_rss_mib(), cap=limit)


def elapsed(fn):
    start = perf_counter()
    value = fn()
    return perf_counter() - start, value


def decision_sample(rounds, seed):
    agents = [create_agent(name, seed=seed) for name in AGENT_NAMES]
    records = []
    for layout, board in cases().items():
        for color in (game.WHITE, game.BROWN):
            state = game.from_legacy(board, color)
            for dice in ((3, 1), (6, 4), (1, 1), (5, 5)):
                for _ in range(rounds):
                    generation, unique = elapsed(lambda: len(game.legal_turns(state, dice)))
                    paths, ordered = elapsed(lambda: sum(1 for _ in _legal_paths(state, dice)))
                    extraction, candidates = elapsed(lambda: list(iter_candidates(state, dice)))
                    decisions, ranking = {}, {}
                    for agent in agents:
                        seconds, chosen = elapsed(lambda: agent.choose_turn(state, dice))
                        decisions[agent.name] = seconds
                        if agent.name != "random":
                            seconds, ranked = elapsed(lambda: agent.select_candidate(candidates))
                            if ranked.turn != chosen:
                                raise AssertionError("ranking precomputed candidates changed the decision")
                            ranking[agent.name] = seconds
                    records.append(dict(layout=layout, dice=dice, unique=unique, ordered=ordered,
                                        generation=generation, paths=paths, extraction=extraction,
                                        decisions=decisions, ranking=ranking))
    count = len(records)
    def ms(values):
        return round(1000 * mean(values), 4)
    policies = {}
    for name in AGENT_NAMES:
        times = [r["decisions"][name] for r in records]
        policies[name] = dict(mean_decision_ms=ms(times), median_decision_ms=1000*median(times),
                              decisions_per_second=count/sum(times),
                              mean_rank_only_ms=ms(r["ranking"][name] for r in records) if name != "random" else None)
    roll_types = {}
    for doubles, label in ((False, "mixed"), (True, "doubles")):
        selected = [r for r in records if (r["dice"][0] == r["dice"][1]) == doubles]
        roll_types[label] = dict(mean_ordered_paths=mean(r["ordered"] for r in selected),
                                 mean_ordered_generation_ms=ms(r["paths"] for r in selected),
                                 mean_generation_plus_features_ms=ms(r["extraction"] for r in selected))
    return dict(positions_and_rolls=count, mean_unique=mean(r["unique"] for r in records),
                mean_ordered=mean(r["ordered"] for r in records),
                mean_legal_generation_ms=ms(r["generation"] for r in records),
                mean_complete_path_generation_ms=ms(r["paths"] for r in records),
                mean_generation_plus_features_ms=ms(r["extraction"] for r in records),
                roll_types=roll_types, policies=policies)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=90)
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if min(args.games, args.rounds, args.max_turns) < 1:
        p.error("games, rounds, and max-turns must be positive")
    report = dict(python=platform.python_version(), platform=platform.platform(), seed=args.seed,
                  simulation=simulation_sample(args.games, args.max_turns, args.seed),
                  decisions=decision_sample(args.rounds, args.seed))
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
