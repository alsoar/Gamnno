"""Shared, actor-relative features of legal complete turns in magnno.

iter_candidates streams ordered paths in canonical order. evaluate_chosen explains
one known-legal Turn without running another legal search. See FEATURES.md.
"""

from math import isfinite
from struct import Struct
from typing import NamedTuple, Optional

import backgammon as game
from heuristic import _legal_paths


_POINTS = Struct("24b")
DEFAULT_BAR_PENALTY = 3


class Side(NamedTuple):
    blots: int
    bar: int
    effective_vulnerability: float
    outside_home_half: int
    distance_to_home_half: int
    leading_checker_distance: int
    bearoff_candidates: int
    eligible: bool
    occupied_points: int
    made_points: int
    stack_concentration: int
    longest_blockade: int
    forward_mobility: float
    threats: int
    nearby_blocks: int
    rear_home_stack: int
    made_mask: int
    blockade_pairs: int
    victory_distance: int


class Hit(NamedTuple):
    step: int
    move: game.Move
    self_hit: bool
    opponent_home: bool
    outside_home: bool
    progress: int
    breaks_eligibility: bool
    last_outsider: bool
    delays_win: bool
    value: int


class Candidate(NamedTuple):
    turn: game.Turn
    own_before: Side
    opp_before: Side
    own_after: Side
    opp_after: Side
    wins: bool
    captures: int
    self_captures: int
    outside_home_captures: int
    opponent_home_captures: int
    capture_value: int
    forward_progress: int
    backward_displacement: int
    backward_moves: int
    backward_distances: tuple
    opponent_backward_distances: tuple
    stack_joins: int
    large_splits: int
    events: tuple

    @property
    def exposure_change(self):
        return self.own_after.blots - self.own_before.blots


class _Step(NamedTuple):
    own_distance: int
    pips: int
    opponent_distance: int
    joins: bool
    splits: bool
    hit: Optional[Hit]


def _penalty(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError("bar_penalty must be a finite, nonnegative number")
    return value


def _side(grid, bar, enemy_bar, penalty):
    # grid is in this color's direction, with its own checkers positive.
    blots = occupied = made = concentration = home_ready = nearby = rear = 0
    run = longest = mask = checkers = 0
    outside, home_distance = bar, bar * 13
    first, last = 24, -1
    for p, n in enumerate(grid):
        if n <= 0:
            run = 0
            continue
        checkers += n
        first = min(first, p)
        last = p
        occupied += 1
        concentration += n * n
        if p < 12:
            outside += n
            home_distance += (12 - p) * n
        elif p >= 18:
            home_ready += n
        if n == 1:
            blots += 1
            run = 0
        else:
            made += 1
            mask |= 1 << p
            run += 1
            longest = max(longest, run)
            nearby += bool((enemy_bar and p >= 18) or min(grid[p + 1:p + 7], default=0) < 0)
            if 12 <= p < 18:
                rear += (18 - p) * n * n

    off = 15 - checkers - bar
    lead = 0 if off else 24 - last
    eligible = outside == 0
    if bar:
        # Re-entry is mandatory. Only the six entry points matter.
        mobility = sum(n >= -1 for n in grid[:6]) / 6
        threats = grid[:6].count(-1)
    else:
        open_faces = threats = 0
        # Mark points geometrically reachable from a friendly source in one die.
        reachable = [False] * 24
        previous = -100
        for p, n in enumerate(grid):
            reachable[p] = p - previous <= 6
            if n <= 0:
                continue
            previous = p
            open_faces += n * sum(v >= -1 for v in grid[p + 1:p + 7])
            if eligible and p >= 18:
                open_faces += n * (1 + (p - 18 if p == first else 0))
        mobility = open_faces / (6 * checkers) if checkers else 0.0
        for p, n in enumerate(grid):
            if n == -1:
                if reachable[p] or any(grid[q] == 0 and reachable[q]
                                       for q in range(p + 1, min(24, p + 7))):
                    threats += 1
            elif n == -2 and reachable[p]:
                # A backward step may split this pair; avoid landing on an own blot.
                if min(grid[p + 1:p + 7], default=1) <= 0:
                    threats += 1
    return Side(blots, bar, blots + penalty * bar, outside, home_distance, lead,
                home_ready if eligible else 0, eligible, occupied, made, concentration,
                longest, mobility, threats, nearby, rear, mask, mask & (mask >> 1),
                25 * outside + home_distance + lead)


def _both(state, player, penalty):
    raw = game.state_key(state)
    grid = _POINTS.unpack_from(raw)
    white = _side(grid, raw[24], raw[25], penalty)
    brown = _side(tuple(-n for n in reversed(grid)), raw[25], raw[24], penalty)
    return (white, brown) if player == game.WHITE else (brown, white)


def side_features(state, player, *, bar_penalty=DEFAULT_BAR_PENALTY):
    """Position features for an explicitly chosen color, independent of side to move."""
    if player not in (game.WHITE, game.BROWN):
        raise ValueError("expected a signed player color")
    return _both(state, player, _penalty(bar_penalty))[0]


def _impacted(raw, move, step, actor, target):
    victim = -move.checker
    board = _POINTS.unpack_from(raw)
    if victim == game.WHITE:
        outside = sum(n for n in board[:12] if n > 0)
        ready = sum(n for n in board[18:] if n > 0)
        progress = move.target
        bar = raw[24]
    else:
        outside = -sum(n for n in board[12:] if n < 0)
        ready = -sum(n for n in board[:6] if n < 0)
        progress = 23 - move.target
        bar = raw[25]
    breaks = not (outside or bar)
    last = outside == 1 and not bar and progress < 12
    delays = breaks and progress >= 18 and ready == 1
    self_hit = move.checker != actor
    value = 0 if self_hit else progress + 12 * breaks + 6 * last + 6 * delays
    return Hit(step, move, self_hit, not self_hit and target < 6, not self_hit and target < 18,
               progress, breaks, last, delays, value)


def _step(raw, move, index, actor):
    # A hit can only occur on a physical point. The wire stores signed counts.
    source, dest = move.source, move.target
    src = -1 if source == game.BAR else (source if actor == game.WHITE else 23 - source)
    dst = 24 if dest == game.OFF else (dest if actor == game.WHITE else 23 - dest)
    hit = None
    if dest != game.OFF and raw[dest] == (-move.checker & 255):
        hit = _impacted(raw, move, index, actor, dst)
    if move.checker != actor:
        return _Step(0, move.die, max(0, src - 11), False, False, hit)
    joins = dest != game.OFF and ((0 < raw[dest] < 128) if actor == game.WHITE else raw[dest] >= 128)
    # Counts are in [-15,15]. Observe the stack before this specific checker leaves.
    splits = source != game.BAR and ((3 <= raw[source] <= 15) if actor == game.WHITE
                                      else (128 <= raw[source] <= 253))
    return _Step(24 - src, dst - src, 0, joins, splits, hit)


def _candidate(turn, before, after, trace):
    own_dist, enemy_dist, events = [], [], []
    forward = displacement = joins = splits = self_hits = outside = opp_home = value = 0
    for step in trace:
        if step.own_distance:
            forward += step.pips
            own_dist.append(step.own_distance)
            joins += step.joins
            splits += step.splits
        else:
            displacement += step.pips
            enemy_dist.append(step.opponent_distance)
        hit = step.hit
        if hit is not None:
            events.append(hit)
            self_hits += hit.self_hit
            outside += hit.outside_home
            opp_home += hit.opponent_home
            value += hit.value
    return Candidate(turn, *before, *after, bool(turn.moves and turn.moves[-1].target == game.OFF),
                     len(events) - self_hits, self_hits, outside, opp_home, value, forward,
                     displacement, len(enemy_dist), tuple(sorted(own_dist, reverse=True)),
                     tuple(sorted(enemy_dist, reverse=True)), joins, splits, tuple(events))


def iter_candidates(state, dice, *, bar_penalty=DEFAULT_BAR_PENALTY, prefix=(), prefer_win=False):
    """Stream full legal paths; prefer_win yields only the first reachable win when present."""
    penalty = _penalty(bar_penalty)
    actor = int(state.player)
    before = _both(state, actor, penalty)
    summaries = {}
    prefix = tuple(prefix)

    def observe(raw, move, index):
        if index < len(prefix):
            move = game.Move(*map(int, move))
        return _step(raw, move, index, actor)

    for turn, trace in _legal_paths(state, dice, prefix, observe_step=observe, prefer_win=prefer_win):
        key = game.state_key(turn.state)
        after = summaries.get(key)
        if after is None:
            after = summaries[key] = _both(turn.state, actor, penalty)
        yield _candidate(turn, before, after, trace)


def evaluate_chosen(state, turn, *, bar_penalty=DEFAULT_BAR_PENALTY):
    """Extract the same features from one known-legal turn without enumerating any alternatives."""
    actor = int(state.player)
    penalty = _penalty(bar_penalty)
    board = game._scratch(state)
    trace = []
    for i, move in enumerate(turn.moves):
        move = game.Move(*map(int, move))
        trace.append(_step(bytes(board), move, i, actor))
        game._push(board, move)
    return _candidate(turn, _both(state, actor, penalty), _both(turn.state, actor, penalty), trace)


def feature_record(candidate):
    """JSON-ready features only. The simulator records compact states and actions separately."""
    result = candidate._asdict()
    del result["turn"]
    for side in ("own_before", "opp_before", "own_after", "opp_after"):
        result[side] = getattr(candidate, side)._asdict()
    result["exposure_change"] = candidate.exposure_change
    result["events"] = [dict(h._asdict(), move=dict(h.move._asdict())) for h in candidate.events]
    return result


# These position accessors are convenient for individual queries. A policy that
# needs several metrics should reuse side_features(...) or the Candidate's Side.
def blots(state, player):
    return side_features(state, player).blots


def effective_vulnerability(state, player, bar_penalty=DEFAULT_BAR_PENALTY):
    return side_features(state, player, bar_penalty=bar_penalty).effective_vulnerability


def exposure_change(before, after, player):
    return blots(after, player) - blots(before, player)


def outside_home_half(state, player):
    return side_features(state, player).outside_home_half


def distance_to_home_half(state, player):
    return side_features(state, player).distance_to_home_half


def leading_checker_distance(state, player):
    return side_features(state, player).leading_checker_distance


def occupied_points(state, player):
    return side_features(state, player).occupied_points


def made_points(state, player):
    return side_features(state, player).made_points


def stack_concentration(state, player):
    return side_features(state, player).stack_concentration


def longest_blockade(state, player):
    return side_features(state, player).longest_blockade


def opponent_forward_mobility(state, player):
    """Forward mobility of the opponent of the supplied player."""
    return side_features(state, -player).forward_mobility


def capture_opportunities(state, player):
    """Number of distinct enemy origins this player can threaten geometrically."""
    return side_features(state, player).threats


def _measured(source, turn_or_player):
    if isinstance(source, Candidate):
        if turn_or_player is not None and turn_or_player != -source.turn.state.player:
            raise ValueError("candidate metrics are relative to its acting player")
        return source
    if isinstance(source, game.State) and isinstance(turn_or_player, game.Turn):
        return evaluate_chosen(source, turn_or_player)
    raise TypeError("supply an evaluated Candidate or an initial State and its legal Turn")


def captures(source, turn_or_player=None):
    return _measured(source, turn_or_player).captures


def self_captures(source, turn_or_player=None):
    return _measured(source, turn_or_player).self_captures


def outside_home_captures(source, turn_or_player=None):
    return _measured(source, turn_or_player).outside_home_captures


def capture_value(source, turn_or_player=None):
    return _measured(source, turn_or_player).capture_value


def total_forward_progress(source, turn_or_player=None):
    return _measured(source, turn_or_player).forward_progress


def backward_displacement(source, turn_or_player=None):
    return _measured(source, turn_or_player).backward_displacement


def immediate_win(turn, player=None):
    """Whether a legal Turn or evaluated Candidate ends in the supplied player's win."""
    if isinstance(turn, Candidate):
        turn = turn.turn
    actor = -int(turn.state.player) if player is None else player
    return game.winner(turn.state) == actor
