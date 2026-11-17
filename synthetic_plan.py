"""Quota-preserving match schedules and independently seeded policy wrappers.

A slot has a fixed allocation stratum and split. Its next attempt advances through
the same color-balanced opponent catalog. Completed concrete matchups can be less
balanced than scheduled ones when their completion probabilities differ.
"""
from collections import Counter
from dataclasses import dataclass, fields
from functools import cached_property, lru_cache
from hashlib import blake2b
from itertools import combinations
import json
from math import isfinite
from random import Random
from typing import Mapping, Optional
from uuid import UUID

from agents import AGENT_NAMES, create_agent


PLAN_VERSION = "magnno-synthetic-plan-v1"
POLICY_VERSION = "magnno-agents-v1"
PURE = tuple(name for name in AGENT_NAMES if name != "random")
DEFAULT_WEIGHTS = (.15, .15, .10, .15, .10, .10, .15, .10, 0.)
HELDOUT_WEIGHTS = (.05, .25, .05, .10, .15, .05, .20, .15, 0.)
SPLITS = ("train", "validation", "test")
CATEGORIES = ("cross_strategy", "pure_selfplay", "turn_mixture", "game_mixture", "exploratory")
CROSS_BUCKETS = ("sprinter_saboteur", "sprinter_hoarder_fortress", "assassin_fortress_blockader",
                 "saboteur_hoarder_blockader", "optimizer_nonrandom", "remaining_nonrandom",
                 "random_nonrandom")
CROSS_PERCENT = (15, 15, 15, 15, 20, 15, 5)


def _digest(items):
    data = json.dumps(items, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return blake2b(data, digest_size=16).digest()


def child_seed(seed, *path):
    """Stable, domain-separated nonnegative integer accepted by SQLite and Random."""
    return int.from_bytes(_digest((seed, *path))[:8], "big") & ((1 << 63) - 1)


def _weights(value):
    if isinstance(value, Mapping):
        if set(value) - set(AGENT_NAMES):
            raise ValueError("mixture_weights contains an unknown policy id")
        value = [value.get(name, 0.) for name in AGENT_NAMES]
    weights = tuple(float(item) for item in value)
    if len(weights) != len(AGENT_NAMES) or not all(isfinite(x) and x >= 0 for x in weights):
        raise ValueError("expected nine nonnegative finite mixture weights")
    total = sum(weights)
    if not isfinite(total) or total <= 0:
        raise ValueError("mixture weights need a positive finite total")
    # Preserve a normalized serialized vector exactly on a config round trip.
    return weights if abs(total - 1.) <= 1e-12 else tuple(x / total for x in weights)


@dataclass(frozen=True)
class PlanConfig:
    target_games: int = 5000
    heldout_games: int = 100
    master_seed: int = 2026
    namespace: str = "pilot"
    max_turns: int = 2000
    max_attempts_per_slot: int = 64
    mixture_weights: tuple = DEFAULT_WEIGHTS
    heldout_weights: tuple = HELDOUT_WEIGHTS
    epsilons: tuple = (.05, .10, .20)

    def __post_init__(self):
        for name, minimum in (("target_games", 0), ("heldout_games", 0),
                              ("max_turns", 1), ("max_attempts_per_slot", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.master_seed) is not int or not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("expected integer master_seed and nonempty namespace")
        for name in ("mixture_weights", "heldout_weights"):
            object.__setattr__(self, name, _weights(getattr(self, name)))
        if self.heldout_games and self.mixture_weights == self.heldout_weights:
            raise ValueError("heldout mixture probabilities must differ from training")
        eps = tuple(float(x) for x in self.epsilons)
        if not eps or len(set(eps)) != len(eps) or not all(0 <= x <= 1 for x in eps):
            raise ValueError("epsilons must be distinct probabilities in [0, 1]")
        object.__setattr__(self, "epsilons", eps)

    def as_dict(self):
        return {f.name: list(value) if isinstance(value, tuple) else value
                for f in fields(self) for value in (getattr(self, f.name),)}

    @cached_property
    def identity(self):
        settings = self.as_dict()
        del settings["max_attempts_per_slot"]
        return _digest((PLAN_VERSION, settings)).hex()


def _config(config):
    if isinstance(config, PlanConfig):
        return config
    values = config.get("plan", config)
    return PlanConfig(**{f.name: values[f.name] for f in fields(PlanConfig) if f.name in values})


def _allocate(total, weights):
    divisor = sum(weights)
    counts = [total * w // divisor for w in weights]
    order = sorted(range(len(weights)), key=lambda i: -(total * weights[i] % divisor))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


def _split_counts(sizes):
    """Integer rounding with exact global column totals and exact stratum sizes."""
    ratios = (8, 1, 1)
    desired = _allocate(sum(sizes), ratios)
    rows = [[n * w // 10 for w in ratios] for n in sizes]
    for col in range(3):
        for _ in range(desired[col] - sum(row[col] for row in rows)):
            candidates = [i for i, row in enumerate(rows) if sum(row) < sizes[i]]
            i = max(candidates, key=lambda k: sizes[k] * ratios[col] - 10 * rows[k][col])
            rows[i][col] += 1
    return rows


@dataclass(frozen=True)
class Slot:
    slot_id: str
    ordinal: int                 # local to allocation stratum, stable across retries
    category: str
    subcategory: str
    split: str


def make_plan(config):
    config = _config(config)
    counts = _allocate(config.target_games, (35, 10, 20, 10, 5))
    cross = _allocate(counts[0], CROSS_PERCENT)
    groups = [(CATEGORIES[0], sub, n) for sub, n in zip(CROSS_BUCKETS, cross)]
    groups.extend((name, "all", n) for name, n in zip(CATEGORIES[1:], counts[1:]))
    distribution = _split_counts([n for _, _, n in groups])
    primary = []
    for (category, sub, n), sizes in zip(groups, distribution):
        labels = [name for name, size in zip(SPLITS, sizes) for _ in range(size)]
        Random(child_seed(config.identity, "primary-split", category, sub)).shuffle(labels)
        for i in range(n):
            primary.append(Slot(f"p-{len(primary):08d}", i, category, sub, labels[i]))
    heldout = [Slot(f"h-{i:08d}", i, "heldout", "all", "heldout_evaluation")
               for i in range(config.heldout_games)]
    slots = primary + heldout
    Random(child_seed(config.identity, "slot-order")).shuffle(slots)
    return slots


@dataclass(frozen=True)
class PolicySpec:
    kind: str
    base: Optional[str] = None
    epsilon: Optional[float] = None
    heldout: bool = False

    @property
    def name(self):
        if self.kind == "pure":
            return self.base
        if self.kind == "noisy":
            return f"noisy:{self.base}:{self.epsilon}"
        return f"{self.kind}_mixture" + (":heldout" if self.heldout else "")


def _cross_catalogs():
    initial = [
        [("sprinter", "saboteur")],
        [("sprinter", x) for x in ("hoarder", "fortress")],
        [("assassin", x) for x in ("fortress", "blockader")],
        [("saboteur", x) for x in ("hoarder", "blockader")],
        [("optimizer", x) for x in PURE if x != "optimizer"],
    ]
    reserved = {frozenset(pair) for group in initial for pair in group}
    initial += [[pair for pair in combinations(PURE, 2) if frozenset(pair) not in reserved],
                [("random", x) for x in PURE]]
    return dict(zip(CROSS_BUCKETS, initial))


_CROSS_PAIRS = _cross_catalogs()


@lru_cache(maxsize=128)
def _catalog(identity, category, sub, epsilons):
    """Interleave wrapper kinds, then alternate white/brown per unordered pair."""
    rng = Random(child_seed(identity, "matchup-assignment", category, sub))
    pure = lambda name: PolicySpec("pure", name)
    turn = PolicySpec("turn", heldout=category == "heldout")
    per_game = PolicySpec("game", heldout=category == "heldout")
    bases = list(PURE)
    rng.shuffle(bases)
    if category == "cross_strategy":
        pairs = [(sub, pure(a), pure(b)) for a, b in _CROSS_PAIRS[sub]]
        rng.shuffle(pairs)
    elif category == "pure_selfplay":
        pairs = [("same_policy", pure(b), pure(b)) for b in bases]
    elif category in ("turn_mixture", "game_mixture"):
        pairs = []
        for b in bases:
            if category == "turn_mixture":
                group = [("turn_vs_turn", turn, turn), ("turn_vs_pure", turn, pure(b))]
            else:
                group = [("game_vs_game", per_game, per_game),
                         ("game_vs_turn", per_game, turn), ("game_vs_pure", per_game, pure(b))]
            rng.shuffle(group)
            pairs.extend(group)
    elif category in ("exploratory", "heldout"):
        pairs = []
        rates = list(epsilons)
        rng.shuffle(rates)
        for row in range(len(rates)):
            for i, b in enumerate(bases):
                rate = rates[(row + i) % len(rates)]
                other = bases[(i + 1 + row % (len(bases) - 1)) % len(bases)]
                noisy = PolicySpec("noisy", b, rate)
                if category == "exploratory":
                    group = [("random_vs_pure", pure("random"), pure(b)),
                             ("random_vs_random", pure("random"), pure("random")),
                             ("noisy_vs_pure", noisy, pure(other))]
                else:
                    other_noisy = PolicySpec("noisy", other, rates[(row + i + 1) % len(rates)])
                    group = [("heldout_turn_vs_game", turn, per_game),
                             ("heldout_noisy_vs_noisy", noisy, other_noisy)]
                rng.shuffle(group)
                pairs.extend(group)
    else:
        raise ValueError(f"unknown allocation {category!r}")
    orientations = {}
    oriented = []
    for label, a, b in pairs * 2:
        key = tuple(sorted((a.name, b.name)))
        if key not in orientations:
            orientations[key] = rng.getrandbits(1)
        left, right = (a, b) if a.name <= b.name else (b, a)
        if orientations[key]:
            left, right = right, left
        orientations[key] ^= 1
        oriented.append((label, left, right))
    return tuple(oriented)


@dataclass(frozen=True)
class GameSpec:
    slot_id: str
    attempt: int
    game_id: str
    generation_seed: int
    dice_seed: int
    sampling_seed: int
    trajectory_seed: int
    white: PolicySpec
    brown: PolicySpec
    category: str
    subcategory: str
    split: str
    mixture_type: str
    max_turns: int

    @property
    def white_agent(self):
        return self.white.name

    @property
    def brown_agent(self):
        return self.brown.name


def spec_for(slot, attempt, config):
    config = _config(config)
    if type(attempt) is not int or not 0 <= attempt < config.max_attempts_per_slot:
        raise ValueError("attempt must be a zero-based index below max_attempts_per_slot")
    catalog = _catalog(config.identity, slot.category, slot.subcategory, config.epsilons)
    label, white, brown = catalog[(slot.ordinal + attempt) % len(catalog)]
    scope = "heldout" if slot.split == "heldout_evaluation" else "primary"
    key = _digest((PLAN_VERSION, config.identity, scope, slot.slot_id, attempt))
    game_id = str(UUID(bytes=key, version=5))
    seed = child_seed(config.master_seed, config.namespace, scope, game_id)
    mix = "+".join(sorted({white.kind, brown.kind}))
    return GameSpec(slot.slot_id, attempt, game_id, seed, child_seed(seed, "dice"),
                    child_seed(seed, "position-sampling"), child_seed(seed, "trajectory-sampling"),
                    white, brown, slot.category, label, slot.split, mix, config.max_turns)


class _Seat:
    """Delegate an entire decision to one registered agent. Streams are seat-local."""
    def __init__(self, descriptor, seed, weights=DEFAULT_WEIGHTS):
        self.descriptor = descriptor
        self.name = descriptor.name
        self.weights = weights
        self._select = Random(child_seed(seed, "strategy-selection"))
        self._explore = Random(child_seed(seed, "exploration-coin"))
        self._actions = Random(child_seed(seed, "random-legal-turn"))
        self._agents = {}
        self.explored = False
        self.game_strategy = None
        self.selected_strategy = descriptor.base

    def _draw(self):
        return self._select.choices(AGENT_NAMES, weights=self.weights, k=1)[0]

    def _next_strategy(self):
        return self.descriptor.base

    def choose_candidate(self, state, dice, rng=None):
        # Ignore simulator-supplied RNG to keep behavior independent of dice and sampling.
        self.selected_strategy = self._next_strategy()
        name = self.selected_strategy
        if name not in self._agents:
            self._agents[name] = create_agent(name, seed=0)
        return self._agents[name].choose_candidate(state, dice, rng=self._actions)

    def choose_turn(self, state, dice, rng=None):
        return self.choose_candidate(state, dice, rng=rng).turn

    @property
    def metadata(self):
        desc = self.descriptor
        data = {"policy_id": self.name, "kind": desc.kind, "base": desc.base,
                "epsilon": desc.epsilon, "agent_version": POLICY_VERSION}
        if desc.kind in ("turn", "game"):
            data["mixture_weights"] = dict(zip(AGENT_NAMES, self.weights))
        if desc.kind == "game":
            data["game_strategy"] = self.game_strategy
        return data


class TurnMixtureAgent(_Seat):
    def _next_strategy(self):
        return self._draw()


class GameMixtureAgent(_Seat):
    def __init__(self, descriptor, seed, weights=DEFAULT_WEIGHTS):
        super().__init__(descriptor, seed, weights)
        self.selected_strategy = self.game_strategy = self._draw()

    def _next_strategy(self):
        return self.game_strategy


class NoisyPolicy(_Seat):
    def _next_strategy(self):
        self.explored = self._explore.random() < self.descriptor.epsilon
        return "random" if self.explored else self.descriptor.base


def build_policies(spec, config):
    """Build two fresh seats for exactly one game, including independent per-game draws."""
    config = _config(config)
    kinds = {"pure": _Seat, "turn": TurnMixtureAgent, "game": GameMixtureAgent, "noisy": NoisyPolicy}
    def seat(color, descriptor):
        weights = config.heldout_weights if descriptor.heldout else config.mixture_weights
        return kinds[descriptor.kind](descriptor, child_seed(spec.generation_seed, color), weights)
    return seat("white", spec.white), seat("brown", spec.brown)
