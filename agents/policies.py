"""Complete-turn policies. Comparison tuples are ordered from most to least important."""

from dataclasses import dataclass, fields
from math import isfinite
from random import Random

import backgammon as game
import features
import heuristic as archived


@dataclass(frozen=True)
class AgentConfig:
    bar_penalty: float = 3
    race_pips: float = 3
    fortress_safety_gain: float = 2
    sabotage_delay: float = 4
    assassin_value_gain: float = 6
    blockade_mobility_gain: float = 0.1
    hoarder_concentration_gain: float = 12
    sacrifice_threshold: float = 0.25
    capture_weight: float = 2.5
    vulnerability_weight: float = 1.0
    forward_weight: float = 0.85
    displacement_weight: float = 0.5
    blockade_weight: float = 1.0
    victory_weight: float = 1.0
    self_capture_weight: float = 1.5

    def __post_init__(self):
        for field in fields(self):
            x = getattr(self, field.name)
            if isinstance(x, bool) or not isinstance(x, (float, int)) or not isfinite(x) or x < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")
        if self.bar_penalty < 1:
            raise ValueError("bar_penalty must be at least one blot per checker")


def dominates(witness, candidate):
    """Both tuples contain primary merit, safety, race components and captures (higher is better)."""
    return all(a >= b for a, b in zip(witness, candidate))


class TurnAgent:
    def __init__(self, config=None, seed=None):
        self.config = config if isinstance(config, AgentConfig) else AgentConfig(**(config or {}))

    def primary(self, c, rank):
        return rank[:1]

    def comparison(self, c, rank):
        a = c.own_after
        return (self.primary(c, rank), -a.effective_vulnerability, -a.outside_home_half,
                -a.distance_to_home_half, -a.leading_checker_distance, c.captures)

    def choose_turn(self, state, dice, rng=None, *, prefix=()):
        return self.choose_candidate(state, dice, rng, prefix=prefix).turn

    def choose_candidate(self, state, dice, rng=None, *, prefix=()):
        prefix = tuple(prefix)
        candidates = features.iter_candidates(state, dice, bar_penalty=self.config.bar_penalty,
                                               prefix=prefix, prefer_win=True)
        return self.select_candidate(candidates, prefix_length=len(prefix))

    def select_candidate(self, candidates, *, prefix_length=0):
        # Most rolls have no sacrifices, or very few compared with ordinary plays.
        # Keep one clean favorite and distinct dominance witnesses. Retain expensive
        # paths only when they could still beat that favorite.
        clean = best = None
        costs = {}
        witnesses = {}
        sacrifices = []
        paid = None if prefix_length else 0
        for i, c in enumerate(candidates):
            if c.wins:
                return c  # Generator order is the canonical mirrored tie-break.
            rank = self.rank(c)
            key = (rank, -i)
            if best is None or key > best[0]:
                best = (key, c)
            if paid is None:
                paid = sum(e.self_hit for e in c.events if e.step < prefix_length)
            cost = c.self_captures - paid
            costs[cost] = max(costs.get(cost, 0), c.opponent_home_captures)
            witness = self.comparison(c, rank)
            witnesses.setdefault(cost, set()).add(witness)
            if not cost:
                if clean is None or key > clean[0]:
                    clean = (key, c)
            elif clean is None or key > clean[0]:
                sacrifices.append((key, c, witness))
        if best is None:
            raise ValueError("there is no complete legal turn for this position and prefix")
        if clean is None or best[0] == clean[0]:
            return best[1]
        # Reaching this point requires a strictly better-ranked sacrifice. A higher
        # canonical tie priority by itself never buys a sacrifice.
        better = [item for item in sacrifices if item[0][0] > clean[0][0]]
        def remaining(c):
            if not prefix_length:
                return c
            # Preserve the complete score and final board for ranking. For eligibility,
            # only new self-hits need payment, and only future events can pay for them.
            return c._replace(self_captures=c.self_captures - paid,
                              events=tuple(e for e in c.events if e.step >= prefix_length))

        clean_continuation = remaining(clean[1])
        for _, c, witness in sorted(better, key=lambda item: item[0], reverse=True):
            continuation = remaining(c)
            if self.admits(continuation, clean_continuation, costs):
                dominated = any(dominates(other, witness) for cost, group in witnesses.items()
                                if cost < continuation.self_captures for other in group)
                if not dominated:
                    return c
        return clean[1]

    def explain(self, state, turn):
        c = features.evaluate_chosen(state, turn, bar_penalty=self.config.bar_penalty)
        if c.wins:
            return "Bore off the winning checker."
        if not turn.moves:
            return "Passed because neither die could be played."
        text = self.description(c)
        if c.self_captures:
            text += f" The turn sent {c.self_captures} friendly checkers to the bar."
        return text


class BaselineAgent(TurnAgent):
    name = "baseline"

    def rank(self, c):
        v = c.own_after.effective_vulnerability
        change = v - c.own_before.effective_vulnerability
        if c.captures and change <= 0:
            if c.outside_home_captures:
                return (4, c.captures, c.outside_home_captures, c.capture_value, -v, c.forward_progress)
            return (3, c.captures, c.capture_value, -v, c.forward_progress)
        return (2 if change < 0 else 1 if change == 0 else 0,
                -v, c.captures, c.forward_progress, c.backward_distances)

    def primary(self, c, rank):
        return rank[0], c.captures, c.outside_home_captures, c.capture_value, c.opponent_home_captures

    def admits(self, c, clean, costs):
        # Each elective self-hit must be followed by a different opponent-home
        # capture, and that many additional captures must be absent on cheaper paths.
        debt = 0
        for event in c.events:
            if event.self_hit:
                debt += 1
            elif event.opponent_home and debt:
                debt -= 1
        return debt == 0 and all(c.opponent_home_captures - home >= c.self_captures - cost
                                 for cost, home in costs.items() if cost < c.self_captures)

    def description(self, c):
        return (f"Captured {c.captures} opposing checkers, including {c.outside_home_captures} outside the home quadrant. "
                f"Vulnerability went from {c.own_before.effective_vulnerability:g} to {c.own_after.effective_vulnerability:g}.")


class SprinterAgent(TurnAgent):
    name = "sprinter"

    def rank(self, c):
        a = c.own_after
        if a.eligible:
            moved_leader = c.own_before.leading_checker_distance in c.backward_distances
            return (1, -a.leading_checker_distance, a.bearoff_candidates, moved_leader, c.forward_progress)
        return (0, -a.outside_home_half, -a.distance_to_home_half, -a.bar, c.backward_distances, c.forward_progress)

    def primary(self, c, rank):
        return rank[:3]

    def admits(self, c, clean, costs):
        a, b, n = c.own_after, clean.own_after, c.self_captures
        if a.outside_home_half != b.outside_home_half:
            return b.outside_home_half - a.outside_home_half >= n
        if a.eligible:
            gain = b.leading_checker_distance - a.leading_checker_distance
        else:
            gain = b.distance_to_home_half - a.distance_to_home_half
        return gain > 0 and gain >= self.config.race_pips * n

    def description(self, c):
        a = c.own_after
        if a.eligible:
            return f"All checkers are home. The leader is {a.leading_checker_distance} pips from bearing off."
        return f"{a.outside_home_half} checkers remain outside the home half, {a.distance_to_home_half} pips from eligibility."


class FortressAgent(TurnAgent):
    name = "fortress"

    def rank(self, c):
        a = c.own_after
        lost = bin(c.own_before.made_mask & ~a.made_mask).count("1")
        return (-a.effective_vulnerability, -c.self_captures, a.made_points, -lost,
                c.captures, c.backward_distances, c.forward_progress)

    def admits(self, c, clean, costs):
        n = c.self_captures
        gain = clean.own_after.effective_vulnerability - c.own_after.effective_vulnerability
        return (gain > 0 and gain >= self.config.fortress_safety_gain * n
                and c.own_after.bar <= clean.own_after.bar and c.captures >= clean.captures + n)

    def description(self, c):
        a = c.own_after
        return f"Finished with {a.blots} blots, {a.bar} on the bar and {a.made_points} protected points."


class SaboteurAgent(TurnAgent):
    name = "saboteur"

    def rank(self, c):
        o = c.opp_after
        bottlenecks = tuple(d for d in c.opponent_backward_distances if d > 0)
        return (c.opp_before.eligible and not o.eligible, o.distance_to_home_half,
                bottlenecks, o.blots, c.capture_value, c.captures, c.backward_displacement)

    def primary(self, c, rank):
        return rank[:2]

    def admits(self, c, clean, costs):
        if c.opp_before.eligible and clean.opp_after.eligible and not c.opp_after.eligible:
            return True
        gain = c.opp_after.distance_to_home_half - clean.opp_after.distance_to_home_half
        return gain > 0 and gain >= self.config.sabotage_delay * c.self_captures

    def description(self, c):
        if c.opp_before.eligible and not c.opp_after.eligible:
            return "Removed the opponent's bearing-off eligibility."
        return (f"Moved opposing checkers backward {c.backward_displacement} pips. "
                f"The opponent needs {c.opp_after.distance_to_home_half} pips to reach its home half.")


class AssassinAgent(TurnAgent):
    name = "assassin"

    def rank(self, c):
        return (c.captures, c.capture_value, c.own_after.threats if not c.captures else 0,
                -c.own_after.effective_vulnerability, c.forward_progress)

    def primary(self, c, rank):
        return rank[:2]

    def admits(self, c, clean, costs):
        n = c.self_captures
        gain = c.capture_value - clean.capture_value
        return c.captures >= clean.captures and gain > 0 and gain >= self.config.assassin_value_gain * n

    def description(self, c):
        if c.captures:
            return f"Captured {c.captures} opposing checkers worth {c.capture_value} capture-value units."
        return f"Finished with {c.own_after.threats} reachable opposing target points."


class BlockaderAgent(TurnAgent):
    name = "blockader"

    def rank(self, c):
        a = c.own_after
        retained = bin(a.blockade_pairs & c.own_before.blockade_pairs).count("1")
        return (a.longest_blockade, -c.opp_after.forward_mobility, retained, a.nearby_blocks,
                c.captures, c.forward_progress)

    def primary(self, c, rank):
        return rank[:2]

    def admits(self, c, clean, costs):
        n = c.self_captures
        new_blocks = c.own_after.longest_blockade - clean.own_after.longest_blockade
        mobility_gain = clean.opp_after.forward_mobility - c.opp_after.forward_mobility
        return (new_blocks >= n and (c.captures >= clean.captures + n or
                mobility_gain > 0 and mobility_gain >= self.config.blockade_mobility_gain * n))

    def description(self, c):
        return (f"The longest blockade spans {c.own_after.longest_blockade} points. "
                f"Estimated opponent forward mobility is {c.opp_after.forward_mobility:.0%}.")


class OptimizerAgent(TurnAgent):
    name = "optimizer"

    def score(self, c):
        w = self.config
        a, b = c.own_after, c.own_before
        block_gain = a.made_points - b.made_points + 2 * (a.longest_blockade - b.longest_blockade)
        return (w.capture_weight * c.capture_value / 24 - w.vulnerability_weight * a.effective_vulnerability / 15
                + w.forward_weight * c.forward_progress / 24 + w.displacement_weight * c.backward_displacement / 24
                + w.blockade_weight * block_gain / 6 + w.victory_weight * (b.victory_distance - a.victory_distance) / 25
                - w.self_capture_weight * c.self_captures)

    def rank(self, c):
        return (self.score(c),)

    def admits(self, c, clean, costs):
        return self.score(c) - self.score(clean) > self.config.sacrifice_threshold

    def description(self, c):
        return (f"Weighted turn score {self.score(c):.3f}, with {c.captures} captures, "
                f"{c.forward_progress} forward pips and {c.backward_displacement} backward pips.")


class HoarderAgent(TurnAgent):
    name = "hoarder"

    def rank(self, c):
        a = c.own_after
        # Each bar checker is a separate unsettled checker for footprint comparisons.
        footprint = a.occupied_points + a.bar
        if a.eligible:
            return (1, -a.leading_checker_distance, a.stack_concentration, -a.effective_vulnerability,
                    a.rear_home_stack, -footprint, -c.large_splits, c.forward_progress)
        return (0, -a.outside_home_half, -a.distance_to_home_half, c.stack_joins, a.rear_home_stack,
                a.stack_concentration, -footprint, -c.large_splits, c.forward_progress)

    def primary(self, c, rank):
        return rank[:3]

    def admits(self, c, clean, costs):
        a, b, n = c.own_after, clean.own_after, c.self_captures
        if b.outside_home_half - a.outside_home_half >= n:
            return True
        if a.eligible and b.eligible and b.leading_checker_distance - a.leading_checker_distance >= max(1, self.config.race_pips) * n:
            return True
        concentration_gain = a.stack_concentration - b.stack_concentration
        return (a.bar <= b.bar and c.captures >= clean.captures + n and concentration_gain > 0
                and concentration_gain >= self.config.hoarder_concentration_gain * n)

    def description(self, c):
        a = c.own_after
        return (f"Finished on {a.occupied_points} occupied points with stack concentration {a.stack_concentration} "
                f"and {a.bar} checkers on the bar.")


class RandomAgent(TurnAgent):
    name = "random"

    def __init__(self, config=None, seed=None):
        super().__init__(config, seed)
        self._rng = Random(seed)

    def choose_turn(self, state, dice, rng=None, *, prefix=()):
        rng = self._rng if rng is None else rng
        if not prefix:
            return game.sample_turn(state, dice, rng)
        distinct = {}
        for turn, _ in archived._legal_paths(state, dice, tuple(prefix)):
            distinct.setdefault(game.state_key(turn.state), turn)
        if not distinct:
            raise ValueError("the prefix has no legal completion")
        return rng.choice(tuple(distinct.values()))

    def explain(self, state, turn):
        return f"Sampled a distinct legal end position uniformly. This turn used {len(turn.moves)} checker moves."

    def choose_candidate(self, state, dice, rng=None, *, prefix=()):
        turn = self.choose_turn(state, dice, rng, prefix=prefix)
        return features.evaluate_chosen(state, turn, bar_penalty=self.config.bar_penalty)
