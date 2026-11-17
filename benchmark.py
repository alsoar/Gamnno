"""Benchmark magnno with reproducible dice and positions. Run python3 benchmark.py."""

import argparse
import itertools
import platform
import random
import sys
from time import perf_counter

import numpy as np

import backgammon as game
from selfplay import DEFAULT_MAX_TURNS


def cases():
    """Starting layout, open contact, bar contention, and a bear-off race."""
    return {
        "opening": game.to_legacy(game.starting_state()),
        "contact": layout(
            {2: 1, 8: 1, 13: 3, 16: 2, 18: 2, 20: 5, 23: 1},
            {7: 3, 9: 3, 11: 2, 14: 4, 22: 1, 25: 2},
        ),
        "bar": layout(
            {0: 2, 10: 2, 13: 4, 18: 3, 20: 4},
            {0: 1, 3: 2, 5: 2, 7: 4, 14: 4, 25: 2},
        ),
        "race": layout(
            {14: 3, 17: 4, 19: 3, 22: 3, 25: 2},
            {2: 2, 4: 3, 7: 3, 10: 3, 13: 4},
        ),
    }


def layout(white, brown):
    return tuple((white.get(i, 0), brown.get(i, 0)) for i in range(27))


def retained_bytes(value):
    """Object sizes for one board/state graph, excluding shared ints and classes.

    For an owning ndarray getsizeof includes its data. Views contribute only
    their header and link to their backing object. Count each identity once.
    """
    seen = set()

    def visit(obj):
        if obj is None or type(obj) in (int, bool) or id(obj) in seen:
            return 0
        seen.add(id(obj))
        size = sys.getsizeof(obj)
        if isinstance(obj, np.ndarray):
            return size + visit(obj.base)
        if isinstance(obj, memoryview):
            return size + visit(obj.obj)
        if isinstance(obj, (bytes, bytearray, np.generic, str)):
            return size
        if isinstance(obj, (tuple, list)):
            return size + sum(visit(item) for item in obj)
        if isinstance(obj, dict):
            # The attribute names are shared by every instance.
            return size + sum(visit(item) for item in obj.values())
        if hasattr(obj, "__dict__"):
            size += visit(vars(obj))
        for cls in type(obj).__mro__:
            slots = vars(cls).get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name not in ("__dict__", "__weakref__") and hasattr(obj, name):
                    size += visit(getattr(obj, name))
        return size

    return visit(value)


def memory_rows():
    state = game.starting_state()
    old_cells = tuple(tuple(cell) for cell in game.to_legacy(state))
    # A legacy tuple board may share equal pairs. Show its best-case sharing too.
    pooled = {}
    shared_cells = tuple(pooled.setdefault(pair, pair) for pair in old_cells)
    return {
        "point array payload": state.board.nbytes,
        "serialized state payload": len(game.serialize_state(state)),
        "compact state retained": retained_bytes(state),
        "legacy (cells, player), as exported": retained_bytes((old_cells, 0)),
        "legacy (cells, player), all equal pairs shared": retained_bytes((shared_cells, 0)),
    }


def generation(rounds, sample=False, seed=2026):
    states = [game.from_legacy(cells, color) for cells in cases().values()
              for color in (game.WHITE, game.BROWN)]
    rolls = list(itertools.product(range(1, 7), repeat=2))
    rng = random.Random(seed)
    # Run one untimed distinct-dice and one doubles call per position.
    for state in states:
        for warmup in ((2, 5), (3, 3)):
            if sample:
                game.sample_turn(state, warmup, rng)
            else:
                game.legal_turns(state, warmup)
    calls = turns = 0
    start = perf_counter()
    for _ in range(rounds):
        for state in states:
            for roll in rolls:
                if sample:
                    game.sample_turn(state, roll, rng)
                    turns += 1
                else:
                    turns += len(game.legal_turns(state, roll))
                calls += 1
    elapsed = perf_counter() - start
    return calls, turns, elapsed


def choose_forward(turns, actor, rng):
    # Terminal first, then maximize number of dice spent advancing our color.
    # Choose uniformly across tied resulting boards.
    best = (-1, -1)
    tied = []
    for turn in turns:
        score = (any(move.target == game.OFF for move in turn.moves),
                 sum(move.checker == actor for move in turn.moves))
        if score > best:
            best = score
            tied = []
        if score == best:
            tied.append(turn)
    return rng.choice(tied)


def selfplay(policy, games, turn_limit, seed):
    wins = {game.WHITE: 0, game.BROWN: 0}
    plies = capped = 0
    start = perf_counter()
    for index in range(games):
        rng = random.Random(seed + index)
        actor, dice = game.opening_roll(rng)
        state = game.starting_state(actor)
        for _ in range(turn_limit):
            if policy == "uniform":
                turn = game.sample_turn(state, dice, rng)
            else:
                turn = choose_forward(game.legal_turns(state, dice), state.player, rng)
            state = turn.state
            plies += 1
            won = game.winner(state)
            if won is not None:
                wins[won] += 1
                break
            dice = game.roll_dice(rng)
        else:
            capped += 1
    elapsed = perf_counter() - start
    return wins[game.WHITE], wins[game.BROWN], capped, plies, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3, help="repetitions of the 288-call generation suite")
    parser.add_argument("--games", type=int, default=6, help="games per policy")
    parser.add_argument("--turn-limit", type=int, default=DEFAULT_MAX_TURNS, help="maximum player turns per game")
    parser.add_argument("--seed", type=int, default=2026, help="first game seed; later games add one")
    args = parser.parse_args()
    if min(args.rounds, args.games, args.turn_limit) < 1:
        parser.error("rounds, games and turn-limit must each be positive")

    print(f"{platform.python_implementation()} {platform.python_version()} | NumPy {np.__version__} | "
          f"{platform.platform()} | seeds {args.seed}..{args.seed + args.games - 1}")
    for name, size in memory_rows().items():
        print(f"{name}: {size} bytes")
    calls, turns, elapsed = generation(args.rounds)
    print(f"Generation: {calls:,} calls, {turns:,} distinct turns in {elapsed:.3f}s; "
          f"{calls / elapsed:,.1f} calls/s, {turns / elapsed:,.1f} distinct turns/s")
    calls, _, elapsed = generation(args.rounds, sample=True, seed=args.seed)
    print(f"Uniform sampling on the same positions: {calls:,} calls in {elapsed:.3f}s; "
          f"{calls / elapsed:,.1f} sampled turns/s")
    for policy in ("forward", "uniform"):
        white, brown, capped, plies, elapsed = selfplay(policy, args.games, args.turn_limit, args.seed)
        print(f"{policy}: {white} white wins, {brown} brown wins, {capped} capped; "
              f"{plies:,} player turns in {elapsed:.3f}s; {plies / elapsed:,.1f} player turns/s, "
              f"{(white + brown) / elapsed:.2f} finished games/s")


if __name__ == "__main__":
    main()
