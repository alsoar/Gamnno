"""Deterministic one-turn policy for magnno. All points use engine coordinates."""

from functools import lru_cache
from typing import NamedTuple, Sequence

import backgammon as game
from backgammon import BAR, OFF, WHITE, Move, State, Turn


class Metrics(NamedTuple):
    """backward_distances are our forward movers' starting distances, largest first."""

    wins: bool
    captures: int
    outside_home_captures: int
    inside_home_captures: int
    capture_value: int
    exposed_before: int
    exposed_after: int
    own_checkers_hit: int
    forward_progress: int
    backward_distances: tuple[int, ...]
    backward_moves: int
    opponent_home_captures: int = 0

    @property
    def exposure_change(self) -> int:
        return self.exposed_after - self.exposed_before


def _oriented(player, point):
    """Map physical points into the moving player's direction; keep BAR/OFF codes."""
    return 23 - point if player != WHITE and point < 24 else point


def _blots(state, player):
    # Count signed +1/-1 only in the board portion of the packed state.
    return game.state_key(state).count(player & 255, 0, 24)


def _measure(player, before, after, moves, hits):
    outside = opponent_home = value = own_hits = forward = backward = 0
    for move in hits:
        if move.checker == player:
            point = _oriented(player, move.target)
            outside += point < 18
            opponent_home += point < 6
            value += 23 - point  # Opponent's advancement from their entry edge.
        else:
            own_hits += 1
    distances = []
    for move in moves:
        if move.checker != player:
            backward += 1
            continue
        distance = 25 if move.source == BAR else 24 - _oriented(player, move.source)
        distances.append(distance)
        forward += distance if move.target == OFF else move.die
    captures = len(hits) - own_hits
    return Metrics(bool(moves and moves[-1].target == OFF), captures, outside,
                   captures - outside, value, before, after, own_hits, forward,
                   tuple(sorted(distances, reverse=True)), backward, opponent_home)


def _hit_steps(state, moves):
    board = game._scratch(state)
    hits = []
    for move in moves:
        _, _, previous = game._push(board, move)
        if previous == -move.checker:
            hits.append(move)
    return tuple(hits)


def evaluate_turn(state: State, turn: Turn) -> Metrics:
    """Explain a known-legal turn. Replay hit events and inspect its final blots."""
    player = int(state.player)
    hits = _hit_steps(state, turn.moves)
    return _measure(player, _blots(state, player), _blots(turn.state, player), turn.moves, hits)


def tier(metrics: Metrics) -> int:
    if metrics.wins:
        return 0
    if metrics.captures and metrics.exposure_change <= 0:
        return 1 if metrics.outside_home_captures else 2
    if metrics.exposure_change < 0:
        return 3
    return 4 if metrics.exposure_change == 0 else 5


def priority_key(m: Metrics) -> tuple:
    """Rank a turn after the roll-wide self-hit eligibility check. Higher is better."""
    level = tier(m)
    if level == 0:
        terms = ()
    elif level == 1:
        terms = (m.captures, m.outside_home_captures, m.capture_value,
                 -m.exposure_change, m.forward_progress)
    elif level == 2:
        terms = (m.captures, m.capture_value, -m.exposure_change, m.forward_progress)
    elif level == 3:
        terms = (-m.exposed_after, m.captures, m.forward_progress)
    elif level == 4:
        # Any level-exposure capture already belongs to tier 1 or 2. Here
        # the explicit racing fallback orders source distances before total pips.
        terms = (m.captures, m.backward_distances, m.forward_progress)
    else:
        terms = (-m.exposure_change, m.captures, m.backward_distances, m.forward_progress)
    return -level, terms, -m.own_checkers_hit


def _legal_paths(state, dice, prefix=(), *, observe_step=None, prefer_win=False):
    """Yield every legal ordered path, its final state and the steps that hit.

    legal_turns handles the full-roll obligations. Its representatives can hide
    differences in where a capture occurred or which checker used a die.
    Expand alternate paths with the engine's own step and transition helpers.
    An optional observer receives (packed pre-step board, move, zero-based
    step index). Its annotations replace the default list of moves that hit.
    With prefer_win, a reachable win yields only the first canonical winning path.
    """
    dice = tuple(dice)
    canonical = game.legal_turns(state, dice)
    if not canonical:
        return
    ends = {game.state_key(t.state): (t.state, game._rank(t.moves)) for t in canonical}
    player = int(state.player)
    pips = game._pips(dice)
    roll_size = len(pips)
    if prefix:
        initial, pips, _ = game._replay(state, pips, prefix)
    else:
        initial = game._scratch(state)
    initial[26] = -player  # Match the side-to-move byte in the canonical children.

    @lru_cache(maxsize=None)
    def branches(position, remaining):
        board = memoryview(bytearray(position)).cast("b")
        children = []
        for move, rest in game._steps(board, player, remaining):
            undo = game._push(board, move)
            if observe_step is None:
                annotation = move if undo[2] == -move.checker else None
            else:
                annotation = observe_step(position, move, roll_size - len(remaining))
            children.append((move, rest, bytes(board), annotation))
            game._pop(board, move, undo)

        def tie_order(child):
            m = child[0]
            return (m.checker != player, _oriented(player, m.source),
                    _oriented(player, m.target), m.die)

        children.sort(key=tie_order)
        return children

    def visit(position, remaining, moves, hits):
        won = bool(moves and moves[-1].target == OFF)
        if remaining and not won:
            children = branches(position, remaining)
            if children:
                for move, rest, child, annotation in children:
                    annotations = hits if annotation is None else hits + (annotation,)
                    yield from visit(child, rest, moves + (move,), annotations)
                return
        end = ends.get(position)
        if end is not None and (won or game._rank(moves) == end[1]):
            yield Turn(moves, end[0]), hits

    @lru_cache(maxsize=None)
    def winning_suffix(position, remaining):
        for move, rest, child, annotation in branches(position, remaining):
            if move.target == OFF:
                return ((move, annotation),), child
            continuation = winning_suffix(child, rest)
            if continuation is not None:
                steps, final = continuation
                return ((move, annotation),) + steps, final
        return None

    try:
        annotations = ()
        if prefix and observe_step is None:
            annotations = _hit_steps(state, prefix)
        elif prefix:
            board = game._scratch(state)
            for i, move in enumerate(prefix):
                annotation = observe_step(bytes(board), move, i)
                if annotation is not None:
                    annotations += (annotation,)
                game._push(board, move)
        initial_key = bytes(initial)
        if prefer_win and not (prefix and prefix[-1].target == OFF):
            if any(t.moves and t.moves[-1].target == OFF for t in canonical):
                win = winning_suffix(initial_key, pips)
                if win is not None:
                    steps, final = win
                    moves = prefix + tuple(move for move, _ in steps)
                    annotations += tuple(note for _, note in steps if note is not None)
                    yield Turn(moves, ends[final][0]), annotations
                    return
        yield from visit(initial_key, pips, prefix, annotations)
    finally:
        branches.cache_clear()
        winning_suffix.cache_clear()
        del winning_suffix
        del visit


def _uncovered_self_hits(player, hits, self_hits):
    """Count the early self-hits left after pairing the latest ones with later captures."""
    credits = 0
    for move in reversed(hits):
        if move.checker != player:
            if credits == 0:
                break
            credits -= 1
            self_hits -= 1
        elif _oriented(player, move.target) < 6:
            credits += 1
    return self_hits


def choose_turn(state: State, dice: Sequence[int], *, prefix=()) -> Turn:
    """Return the best complete legal turn, optionally keeping steps already played."""
    prefix = tuple(prefix)
    player = int(state.player)
    before = _blots(state, player)
    # A sacrifice is justified relative to the alternatives on this roll. Keep
    # only the best path for each combination of counts needed for that comparison.
    best_by_profile = {}
    for order, (turn, hits) in enumerate(_legal_paths(state, dice, prefix)):
        metrics = _measure(player, before, _blots(turn.state, player), turn.moves, hits)
        cost = metrics.own_checkers_hit
        uncovered = _uncovered_self_hits(player, hits, cost) if cost else 0
        profile = (cost, metrics.opponent_home_captures, uncovered)
        rank = (priority_key(metrics), -order)
        prior = best_by_profile.get(profile)
        if prior is None or rank > prior[0]:
            best_by_profile[profile] = (rank, turn)
    if not best_by_profile:
        if prefix:
            raise ValueError("prefix cannot complete a legal turn")
        raise ValueError("the game has already finished")

    home_by_cost = {}
    for cost, home, _ in best_by_profile:
        home_by_cost[cost] = max(home_by_cost.get(cost, 0), home)
    forced = min(home_by_cost)
    best = best_rank = None
    for (cost, home, uncovered), (rank, turn) in best_by_profile.items():
        wins = bool(turn.moves and turn.moves[-1].target == OFF)
        if not wins and cost > forced:
            # Each optional self-hit needs a distinct subsequent opponent-home
            # capture, and one more such capture than cheaper alternatives per
            # extra hit. Events in a fixed prefix are common to all alternatives.
            if uncovered > forced:
                continue
            if any(home - other_home < cost - other_cost
                   for other_cost, other_home in home_by_cost.items() if other_cost < cost):
                continue
        if best_rank is None or rank > best_rank:
            best, best_rank = turn, rank
    return best
