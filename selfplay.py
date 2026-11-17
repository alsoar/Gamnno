"""Run magnno heuristic self-play. Example: python3 selfplay.py --games 10 --seed 2026"""

import argparse
from collections import Counter
import json
from random import Random
from typing import NamedTuple, Optional

import backgammon as game
from game_state import _integer
import heuristic


DEFAULT_SEED = 2026
DEFAULT_MAX_TURNS = 2000


class GameResult(NamedTuple):
    """Per-game totals up to a win or the turn limit. An unfinished winner is None."""

    winner: Optional[int]
    turns: int
    passes: int
    checker_moves: int
    captures: int
    self_hits: int
    backward_moves: int
    exposed_sum: int
    final_state: game.State


def play_game(agent_white, agent_brown, seed=None, max_turns=DEFAULT_MAX_TURNS) -> GameResult:
    """Play at most max_turns individual player turns, including the opening and passes.

    Each agent is a callable (state, dice) -> legal Turn, or an object with
    choose_turn(state, dice). Use the resulting immutable Turn.state directly.
    Dice come from this game's own RNG. An omitted seed uses DEFAULT_SEED.
    A win on the final allowed turn counts.
    """
    limit = _integer(max_turns, "max_turns")
    if limit < 1:
        raise ValueError("max_turns must be positive")
    agents = {game.WHITE: getattr(agent_white, "choose_turn", agent_white),
              game.BROWN: getattr(agent_brown, "choose_turn", agent_brown)}
    if not all(callable(agent) for agent in agents.values()):
        raise TypeError("expected callable agents or objects with choose_turn(state, dice)")
    rng = Random(DEFAULT_SEED if seed is None else _integer(seed, "seed"))
    player, dice = game.opening_roll(rng)
    state = game.starting_state(player)
    passes = moves = captures = self_hits = backward = exposed = 0
    for turns in range(1, limit + 1):
        turn = agents[state.player](state, dice)
        events = heuristic.evaluate_turn(state, turn)
        passes += int(not turn.moves)
        moves += len(turn.moves)
        captures += int(events.captures)
        self_hits += int(events.own_checkers_hit)
        backward += int(events.backward_moves)
        exposed += int(events.exposed_after)
        state = turn.state
        winning_color = game.winner(state)
        if winning_color is not None or turns == limit:
            return GameResult(winning_color, turns, passes, moves, captures,
                              self_hits, backward, exposed, state)
        dice = game.roll_dice(rng)


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def _duration(histogram):
    """Compute an exact mean and median without storing a list of game histories."""
    histogram = dict(sorted(histogram.items()))
    games = sum(histogram.values())
    total = sum(length * count for length, count in histogram.items())
    median = None
    if games:
        lower, upper = (games - 1) // 2, games // 2
        seen = 0
        for length, count in histogram.items():
            if seen <= lower < seen + count:
                left = length
            if seen <= upper < seen + count:
                median = (left + length) / 2
                break
            seen += count
    return {"mean": _fraction(total, games), "median": median, "histogram": histogram}


def simulate_games(n_games, agent_white, agent_brown, seed=DEFAULT_SEED, max_turns=DEFAULT_MAX_TURNS):
    """Aggregate games online. Game i (zero-based) uses seed + i.

    Captures and self-hits are means over all starts, through game end or the cap.
    Exposure is the moving player's final on-board blot count averaged over all
    observed player turns, including passes. Backward fraction weights individual
    checker moves equally. Undefined denominators produce None (JSON null).
    """
    n_games = _integer(n_games, "n_games")
    seed = _integer(seed, "seed")
    limit = _integer(max_turns, "max_turns")
    if n_games < 0 or limit < 1:
        raise ValueError("n_games must be nonnegative and max_turns must be positive")
    finished_lengths, capped_lengths = Counter(), Counter()
    white = brown = turns = passes = steps = captures = self_hits = backward = exposed = 0
    for index in range(n_games):
        result = play_game(agent_white, agent_brown, seed + index, limit)
        if result.winner is None:
            capped_lengths[result.turns] += 1
        else:
            finished_lengths[result.turns] += 1
            white += int(result.winner == game.WHITE)
            brown += int(result.winner == game.BROWN)
        turns += result.turns
        passes += result.passes
        steps += result.checker_moves
        captures += result.captures
        self_hits += result.self_hits
        backward += result.backward_moves
        exposed += result.exposed_sum

    completed = white + brown
    return {
        "seed": seed, "max_turns": limit, "games": n_games,
        "completed": completed, "unfinished": n_games - completed,
        "white_wins": white, "brown_wins": brown,
        "win_rate_all_starts": {"white": _fraction(white, n_games), "brown": _fraction(brown, n_games)},
        "win_rate_completed": {"white": _fraction(white, completed), "brown": _fraction(brown, completed)},
        "length_in_player_turns": {
            "completed": _duration(finished_lengths),
            "unfinished_observed": _duration(capped_lengths),
            "all_observed": _duration(finished_lengths + capped_lengths),
        },
        "turns": turns, "passes": passes, "checker_moves": steps,
        "captures": captures, "self_hits": self_hits, "backward_moves": backward,
        "captures_per_game_started": _fraction(captures, n_games),
        "self_hits_per_game_started": _fraction(self_hits, n_games),
        "acting_blots_after_per_turn": _fraction(exposed, turns),
        "backward_fraction_of_checker_moves": _fraction(backward, steps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10, help="games to play")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for game zero")
    parser.add_argument("--turn-limit", "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                        help="maximum individual player turns per game")
    args = parser.parse_args()
    if args.games < 0 or args.turn_limit < 1:
        parser.error("games must be nonnegative and turn-limit must be positive")
    report = simulate_games(args.games, heuristic.choose_turn, heuristic.choose_turn,
                            seed=args.seed, max_turns=args.turn_limit)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
