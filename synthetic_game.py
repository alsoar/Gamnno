"""One validated synthetic game, pre-roll position sampling, and compact trajectories.

Sampling never sees dice, actions, the acting policy's choice, or the winner.
The opening roll sets both the opener and its dice, so turn 1 is in the game
trace only. All later sampling callbacks execute before the simulator rolls.
"""

from dataclasses import dataclass, field
from hashlib import blake2b
from heapq import heappush, heapreplace
from math import floor, isfinite
from random import Random
from struct import Struct
from typing import NamedTuple, Optional
import zlib

import backgammon as game
from agents import AGENT_NAMES
from simulation import _play
from training import TrainingRecord, deserialize_record, serialize_record


PHASES = ("early", "middle", "late", "near_terminal")
EXPLORED = 128
_POLICY_IDS = {name: i for i, name in enumerate(AGENT_NAMES)}
_POINTS = Struct("24b")
_HEADER = Struct("<4sIB")
_MAGIC = b"MGT\x01"


@dataclass(frozen=True)
class PhaseThresholds:
    early_outside: int = 6
    early_distance: int = 24
    late_outside: int = 2
    near_outside: int = 1
    near_distance: int = 2
    near_lead: int = 6

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= self.near_outside <= self.late_outside < self.early_outside <= 15:
            raise ValueError("outside thresholds must follow near <= late < early <= 15")
        if self.near_lead > 24:
            raise ValueError("near_lead must be <= 24")


@dataclass(frozen=True)
class SamplingConfig:
    positions_per_game: int = 15
    phase_weights: tuple = (0.20, 0.30, 0.30, 0.20)
    phase_thresholds: PhaseThresholds = field(default_factory=PhaseThresholds)
    min_turn_gap: int = 3

    def __post_init__(self):
        for name in ("positions_per_game", "min_turn_gap"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        weights = self.phase_weights
        if isinstance(weights, dict):
            if set(weights) != set(PHASES):
                raise ValueError("phase_weights must specify all four phases")
            weights = [weights[key] for key in PHASES]
        weights = tuple(float(w) for w in weights)
        if len(weights) != 4 or not all(isfinite(w) and w >= 0 for w in weights):
            raise ValueError("expected four nonnegative finite phase weights")
        total = sum(weights)
        if not isfinite(total) or total <= 0:
            raise ValueError("phase weights must have positive finite total")
        object.__setattr__(self, "phase_weights", tuple(w / total for w in weights))
        limits = self.phase_thresholds
        if isinstance(limits, dict):
            limits = PhaseThresholds(**limits)
        if not isinstance(limits, PhaseThresholds):
            raise ValueError("expected PhaseThresholds or a threshold mapping")
        object.__setattr__(self, "phase_thresholds", limits)


_DEFAULT_LIMITS = PhaseThresholds()


def classify_phase(state, thresholds=None):
    """Return the PHASES index. Distances use each color's physical direction.

    A bar checker is an outsider 13 pips from the home half. Near takes priority
    when either side is eligible, or has no bar, <=1 outsider, <=2 total pips to
    eligibility and a leader <=6 pips from the exit. Late requires either side
    to have <=2 outsiders. Early requires BOTH sides to have >=6 outsiders and
    >=24 pips to eligibility. The remaining positions are middle. These are
    configurable geometric proxies.
    """
    limits = _DEFAULT_LIMITS if thresholds is None else thresholds
    if isinstance(limits, dict):
        limits = PhaseThresholds(**limits)
    raw = game.state_key(state)
    board = _POINTS.unpack_from(raw)
    early = True
    late = False
    for points, bar in ((board, raw[24]), (tuple(-n for n in reversed(board)), raw[25])):
        outside = bar
        distance = 13 * bar
        for index, count in enumerate(points[:12]):
            if count > 0:
                outside += count
                distance += (12 - index) * count
        if outside == 0:
            return 3
        if (not bar and outside <= limits.near_outside and distance <= limits.near_distance
                and any(n > 0 for n in points[24 - limits.near_lead:])):
            return 3
        late |= outside <= limits.late_outside
        early &= outside >= limits.early_outside and distance >= limits.early_distance
    return 2 if late else 0 if early else 1


class _Position(NamedTuple):
    rank: int
    turn: int
    key: bytes
    phase: int
    player: int


class PositionSampler:
    """Bounded bottom-K sampling of distinct states per phase and side to move.

    Each state has one seeded rank throughout the game. An evicted state cannot
    displace newer, better ranks on a later visit. At most 16 * target states
    are retained, independent of game length. The draw balances colors, uses a
    preferred turn gap, and backfills shortages from other distinct positions.
    """
    def __init__(self, seed, config):
        self.config = config
        self.salt = blake2b(str(seed).encode("ascii"), digest_size=16).digest()
        tie_rng = Random(seed)
        self.quota_ties = tuple(tie_rng.random() for _ in PHASES)
        self.phase_visits = [0] * 4
        self.buckets = {}
        self.capacity = 2 * config.positions_per_game

    def offer(self, number, state):
        phase = classify_phase(state, self.config.phase_thresholds)
        self.phase_visits[phase] += 1
        key = game.state_key(state)
        player = int(state.player)
        group = (phase, player)
        if group not in self.buckets:
            self.buckets[group] = {}, []
        kept, heap = self.buckets[group]
        if key in kept:
            return
        digest = blake2b(key, key=self.salt, digest_size=12).digest()
        rank = int.from_bytes(digest + key, "big")
        item = (-rank, key)
        if len(heap) == self.capacity:
            if rank >= -heap[0][0]:
                return
            del kept[heapreplace(heap, item)[1]]
        else:
            heappush(heap, item)
        kept[key] = _Position(rank, number, key, phase, player)

    def select(self):
        quota = [self.config.positions_per_game * w for w in self.config.phase_weights]
        counts = list(map(floor, quota))
        remainders = sorted(range(4), key=lambda p: (counts[p] - quota[p], self.quota_ties[p]))
        for phase in remainders[:self.config.positions_per_game - sum(counts)]:
            counts[phase] += 1
        pending = [row for kept, _ in self.buckets.values() for row in kept.values()]
        chosen = []
        phase_count = [0] * 4
        side_count = {game.WHITE: 0, game.BROWN: 0}
        # First respect phase targets. Reallocate absent phase slots only after
        # testing all retained distinct states, including neighboring turns.
        for enforce_quota in (True, False):
            for gap in dict.fromkeys((self.config.min_turn_gap, 1)):
                while pending and len(chosen) < self.config.positions_per_game:
                    available = (r for r in pending
                                 if (not enforce_quota or phase_count[r.phase] < counts[r.phase])
                                 and all(abs(r.turn - other.turn) >= gap for other in chosen))
                    row = min(available, key=lambda r: (side_count[r.player], r.rank), default=None)
                    if row is None:
                        break
                    pending.remove(row)
                    chosen.append(row)
                    phase_count[row.phase] += 1
                    side_count[row.player] += 1
        return sorted(chosen, key=lambda r: r.turn)


class Ply(NamedTuple):
    number: int
    state: game.State
    dice: tuple
    moves: tuple
    after: game.State
    policy_code: int
    captures: int
    self_captures: int


def policy_identity(code):
    """Return (pure registry name, epsilon exploration flag) for a trace byte."""
    if code < 0 or code > 255 or (code & ~EXPLORED) >= len(AGENT_NAMES):
        raise ValueError("invalid pure-policy code")
    return AGENT_NAMES[code & ~EXPLORED], bool(code & EXPLORED)


def iter_trajectory(blob):
    """Decode a zlib-compressed MGT v1 game, including the winning or capped ply.

    The payload begins with four-byte magic, little-endian uint32 ply count, winner byte
    (0 capped, 1 white, 2 brown), 27-byte opening state. Each ply stores one
    pure-policy byte, packed hit counts (low nibble opponent, high own), the
    existing training codec's roll/flags/move words, and the 27-byte afterstate.
    """
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(blob) + decoder.flush()
    except zlib.error as exc:
        raise ValueError("invalid compressed trajectory") from exc
    if not decoder.eof or decoder.unused_data or len(raw) < _HEADER.size + game.STATE_SIZE:
        raise ValueError("truncated or trailing trajectory data")
    magic, turns, win = _HEADER.unpack_from(raw)
    if magic != _MAGIC or not turns or win > 2:
        raise ValueError("invalid trajectory header")
    start = _HEADER.size
    state = game.deserialize_state(raw[start:start + game.STATE_SIZE])
    if state != game.starting_state(int(state.player)):
        raise ValueError("invalid trajectory opening")
    start += game.STATE_SIZE
    for number in range(1, turns + 1):
        if start + 4 > len(raw):
            raise ValueError("truncated trajectory turn")
        policy, hits, _, length = raw[start:start + 4]
        policy_identity(policy)
        if length > 4 or (hits & 15) + (hits >> 4) > length:
            raise ValueError("invalid trajectory flags or capture counts")
        end = start + 4 + 2 * length
        record = deserialize_record(game.state_key(state) + raw[start + 2:end])
        if number == 1:
            white_die, brown_die = record.dice
            if white_die == brown_die or (white_die > brown_die) != (state.player == game.WHITE):
                raise ValueError("trajectory opening disagrees with the opening dice")
        after = game.deserialize_state(raw[end:end + game.STATE_SIZE])
        if after.player != -state.player:
            raise ValueError("trajectory turn order mismatch")
        if number < turns and game.winner(after) is not None:
            raise ValueError("trajectory continues after a win")
        yield Ply(number, state, record.dice, record.action, after, policy, hits & 15, hits >> 4)
        state = after
        start = end + game.STATE_SIZE
    if start != len(raw) or game.winner(state) != (None, game.WHITE, game.BROWN)[win]:
        raise ValueError("trajectory length or terminal winner mismatch")


class _Observer:
    def __init__(self, sampling, seed, archive):
        self.sampler = PositionSampler(seed, sampling)
        self.trace = bytearray()
        self.plies = bytearray() if archive else None
        self.initial = self.current = None

    def opened(self, state):
        self.initial = self.current = game.state_key(state)

    def before_roll(self, number, state):
        # Sample the last verified afterstate before the next die is drawn.
        if self.current != game.state_key(state) or number != len(self.trace) + 1:
            raise ValueError("pre-roll callback does not follow the verified turn")
        self.sampler.offer(number, state)

    def chosen(self, number, state, dice, policy, candidate):
        if self.current != game.state_key(state) or number != len(self.trace) + 1:
            raise ValueError("recorded action does not follow the previous state")
        turn = candidate.turn
        if game.apply_turn(state, dice, turn.moves) != turn.state:
            raise ValueError("policy afterstate disagrees with engine replay")
        self.current = game.state_key(turn.state)
        code = _POLICY_IDS[policy.selected_strategy] | (EXPLORED if policy.explored else 0)
        self.trace.append(code)
        if self.plies is not None:
            record = serialize_record(TrainingRecord(state, dice, turn.moves))
            self.plies.extend((code, candidate.captures | (candidate.self_captures << 4)))
            self.plies.extend(record[game.STATE_SIZE:])
            self.plies.extend(self.current)

    def compact_trajectory(self, turns, winner):
        if self.plies is None:
            return None
        header = _HEADER.pack(_MAGIC, turns, (None, game.WHITE, game.BROWN).index(winner))
        return zlib.compress(header + self.initial + self.plies, level=1)


@dataclass
class SyntheticGame:
    opener: int
    winner: Optional[int]
    turns: int
    initial: bytes
    final: bytes
    samples: list
    trajectory: Optional[bytes]
    policy_trace: bytes
    white_meta: dict
    brown_meta: dict
    stats: dict


def simulate(spec, plan_config, sampling_config=None, record_trajectory=False):
    """Play one independently seeded attempt with the project's complete-turn simulator.

    Samples are (turn, 27-byte state, phase index, selected strategy code).
    Outcome is intentionally omitted here; the dataset writer labels only these
    completed-game samples, by comparing the side to move with this game's winner.
    """
    from synthetic_plan import build_policies

    sampling = SamplingConfig(**(sampling_config or {})) if isinstance(sampling_config, dict) else sampling_config
    sampling = SamplingConfig() if sampling is None else sampling
    if not isinstance(sampling, SamplingConfig):
        raise ValueError("expected SamplingConfig or a sampling configuration mapping")
    config_cap = (plan_config.get("max_turns", 2000) if isinstance(plan_config, dict)
                  else getattr(plan_config, "max_turns", 2000))
    cap = game._integer(getattr(spec, "max_turns", config_cap), "max_turns")
    if cap < 1:
        raise ValueError("max_turns must be positive")
    white, brown = build_policies(spec, plan_config)
    observe = _Observer(sampling, spec.sampling_seed, record_trajectory)
    result = _play(white, brown, spec.dice_seed, cap, spec.game_id, 0, observer=observe)
    final = game.deserialize_state(observe.current)
    if result.winner != game.winner(final) or result.turns != len(observe.trace):
        raise ValueError("simulator result disagrees with verified moves")
    rows = []
    if result.winner is not None:
        for row in observe.sampler.select():
            state = game.deserialize_state(row.key)
            if game.winner(state) is not None or state.player != row.player:
                raise ValueError("invalid primary value position")
            rows.append((row.turn, row.key, row.phase, observe.trace[row.turn - 1]))
    return SyntheticGame(int(game.deserialize_state(observe.initial).player), result.winner,
                         result.turns, observe.initial, observe.current, rows,
                         observe.compact_trajectory(result.turns, result.winner),
                         bytes(observe.trace), white.metadata, brown.metadata,
                         {"white": dict(result.white_stats), "brown": dict(result.brown_stats),
                          "phase_visits": observe.sampler.phase_visits})
