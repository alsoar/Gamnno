# Playing policies

The registry provides nine agents. The eight deterministic policies look only at the current roll. They share the game's complete-turn generator and the same positional and event features. The original policy in `heuristic.py` remains available to reproduce the archived 100-game experiment. The registry's `baseline` uses the current capture-value formula and bar-aware vulnerability.

```python
from agents import AGENT_NAMES, AgentConfig, create_agent
from backgammon import starting_state

state = starting_state()
agent = create_agent("assassin", {"assassin_value_gain": 8})
turn = agent.choose_turn(state, (2, 3))
print(turn.moves, agent.explain(state, turn))
next_state = turn.state
```

Use `choose_candidate` in simulations to obtain the same choice with its already computed features. Both interfaces accept an optional `rng` and keyword-only `prefix` of moves already played on the roll. A prefix constrains the choice to complete legal continuations and is included in the returned turn. Each agent has `.name` and an immutable `.config`. Configuration accepts either a mapping or an `AgentConfig` instance. The feature definitions and capture-value units are described in [FEATURES.md](FEATURES.md).

## Comparing complete turns

Dice order, hits and the locations of pieces when moved can distinguish sequences ending on the same board. Deterministic agents consider every legal ordered sequence consistent with the prefix. Event features retain those distinctions. Positional features use the resulting state. Before other criteria, an available victory selects the first canonical winning sequence.

The engine's canonical ordering compares each step by friendly movement first, then by source, destination and die in the acting player's orientation. A closer tie throughout the full decision tree goes to the earlier complete sequence. Reversing the board and exchanging colors preserves decisions.

Each row lists criteria in descending priority. A criterion to the right resolves ties on all earlier criteria.

| Agent | Decision after checking for a win |
| --- | --- |
| **Baseline** | First consider captures outside the six-point home quadrant with no increase in vulnerability. Maximize captures, outside captures and capture value, then minimize vulnerability and maximize friendly progress. Next consider other captures with no increase in vulnerability, using the same order without the outside-capture criterion. Otherwise reduce vulnerability, maintain it if reduction is unavailable, or minimize the increase. Within each fallback tier prefer lower vulnerability, more captures, more forward pips and then the most backward friendly sources. |
| **Sprinter** | Prefer reaching home-half eligibility. While still outside, minimize the number outside, distance to the boundary and friendly bar count. Prefer advancing the most backward sources and then more forward pips. Once eligible, minimize the leader's distance to bearing off, maximize checkers within one exact die of the exit, prefer moving from the initial leader's distance, and maximize forward progress. |
| **Fortress** | Minimize effective vulnerability and self-hits. Maximize made points, minimize original made points lost, then prefer more captures, advancing the most backward friendly sources and more forward pips. A made point contains two or more checkers. |
| **Saboteur** | Break existing enemy eligibility, maximize the enemy's distance to eligibility, and prefer pushing checkers already farthest outside their home half. Compare those positive source distances largest first. Then maximize enemy blots, capture value, captures and total enemy backward pips. |
| **Assassin** | Maximize captures and capture value. When capture count is zero, maximize reachable enemy target points. Then minimize vulnerability and maximize friendly progress. An extra blot can be acceptable in exchange for a better capture. |
| **Blockader** | Maximize the longest contiguous run of made points, minimize enemy forward mobility, preserve adjacent pairs from the original blockade, and favor made points near opposing checkers. Then maximize captures and friendly forward pips. |
| **Optimizer** | Maximize the weighted change in position given below, subject to the sacrifice threshold. |
| **Hoarder** | While outside the home half, minimize the number outside and their distance to entry. Prefer friendly landings onto occupied friendly points, large stacks near the rear home boundary, and larger total concentration. Minimize footprint and departures from stacks of three or more, then maximize friendly progress. Once all are home, the leader's distance to the exit takes precedence over concentration. Next prefer higher concentration, lower vulnerability, rear-home stacks, smaller footprint, fewer large-stack departures and forward progress. |
| **Random** | Sample uniformly from distinct legal resulting states. Each state receives one entry regardless of how many move orders reach it. Wins, passes and self-hits participate under the same rule. A pass occurs only when required by the engine. |

The home half covers 12 points. The home quadrant covers six. Distance and source rankings are player-relative. The bar counts as outside the half and carries re-entry distance. Friendly checker concentration is the sum of squared on-board stack sizes. Hoarder's footprint counts occupied points plus each bar checker, and its rear-stack measurement favors made points nearest entry into the home half.

## When a self-hit is worth taking

The policies count each friendly checker sent to the bar, including a checker re-entered later on the same turn. Default effective vulnerability is friendly blots plus three times friendly bar occupancy. Set `bar_penalty` to change the cost. The minimum is one, so simply transferring a blot to the bar cannot lower this measure.

When a self-hit-free line exists, a sacrifice must strictly outrank the best such line under the agent's priorities and pass its tactical test below. A sacrifice is also dominated if a legal line with fewer self-hits is at least as good on **all** of these comparisons:

- Its agent-specific primary criterion.
- Lower effective vulnerability.
- Fewer friendly checkers outside the home half, less distance to eligibility, and a closer leading checker. These are three separate comparisons.
- At least as many opposing captures.

For dominance, Baseline compares its priority tier and capture metrics, including opponent-home captures. Sprinter and Hoarder compare the first three criteria of their current branch. Fortress compares vulnerability. Saboteur compares eligibility disruption and enemy distance. Assassin compares capture count and value. Blockader compares run length and enemy mobility. Optimizer compares its own numerical score.

The tests below compare against the best self-hit-free line for that same agent and roll. Here *n* is the number of elective self-hits in the candidate. With a fixed prefix, costs and subsequent capture credits for the gate start at the first unplayed step. An earlier self-hit already incurred by a human cannot exempt an additional sacrifice. Ranking and reporting still measure the whole turn. When every legal continuation requires a self-hit, choose the normally highest-ranked legal continuation. Immediate victories take precedence throughout.

| Agent | Required additional benefit |
| --- | --- |
| **Baseline** | Each elective hit needs a different **later** opposing capture in the opponent's six-point home quadrant. Against every available lower-cost class, the line must gain at least one additional opponent-home capture for each extra self-hit. |
| **Sprinter** | Bring at least *n* additional checkers home, or, with equal numbers outside, save at least `race_pips × n` pips to eligibility. When both lines finish eligible, compare the leader's distance instead. |
| **Fortress** | Lower vulnerability by at least `fortress_safety_gain × n`, leave no extra friendly bar checkers, and capture at least *n* additional opponents. |
| **Saboteur** | Break opponent eligibility when the clean line preserves it, or add at least `sabotage_delay × n` pips to the opponent's distance from eligibility. |
| **Assassin** | Equal or exceed the clean capture count and gain at least `assassin_value_gain × n` capture-value units. A higher-value capture can justify a sacrifice even when capture count stays equal. |
| **Blockader** | Lengthen the blockade by at least *n* extra points, and either gain *n* additional captures or reduce enemy mobility by at least `blockade_mobility_gain × n`. Mobility is a 0–1 open-destination estimate. |
| **Hoarder** | Bring at least *n* additional checkers home, or when eligible improve leader distance by at least `max(1, race_pips) × n`. A consolidation sacrifice must gain at least `hoarder_concentration_gain × n` concentration and *n* extra captures while leaving no extra bar checkers. |
| **Optimizer** | The score after charging for the self-hit must beat the best clean score by **more than** `sacrifice_threshold`. |

For example, put 14 White checkers on point 10 and one on 12. Put eight Brown checkers on 9, one on 3, and six on 14. With a 3 and a 4, Assassin moves a Brown checker from 9 to 12, hitting White's blot. White re-enters on 3 and captures the Brown blot there. This line's capture value is 20. The best clean capture is worth 10. With the default margin of 6, Assassin takes the sacrifice. Fortress avoids it.

## Optimizer units and settings

Optimizer sums seven signed contributions. The fixed divisors scale a pip, blot or structural improvement consistently across positions. The score can exceed one.

| Contribution | Input | Coefficient |
| --- | --- | --- |
| Opponent captures | Total capture value / 24 | `capture_weight = 2.5` |
| Friendly risk | −Final effective vulnerability / 15 | `vulnerability_weight = 1.0` |
| Friendly advance | Actual forward pips / 24 | `forward_weight = 0.85` |
| Enemy displacement | Backward pips / 24 | `displacement_weight = 0.5` |
| Blocking improvement | (Change in made points + 2 × change in longest blockade) / 6 | `blockade_weight = 1.0` |
| Victory proximity | Decrease in (25 × outside count + distance to eligibility + leader distance) / 25 | `victory_weight = 1.0` |
| Sacrifice cost | −Number of friendly self-hits | `self_capture_weight = 1.5` |

The default tactical gates are `race_pips=3`, `fortress_safety_gain=2`, `sabotage_delay=4`, `assassin_value_gain=6`, `blockade_mobility_gain=0.1`, `hoarder_concentration_gain=12`, and `sacrifice_threshold=0.25`. Gains in the compared tactical metric must be positive even with its gate set to zero. Coefficients and gates accept finite nonnegative numbers.

```python
from agents import create_agent
from random import Random

cautious = create_agent("optimizer", {"self_capture_weight": 2, "sacrifice_threshold": 0.5})
sampler = create_agent("random", seed=73)
# Reuse sampler to advance its seeded sequence, or pass a simulation-owned RNG.
sampled_turn = sampler.choose_turn(state, (1, 5), rng=Random(2026))
```

Run the policy tests with `python -m unittest test_agents`. The tests exercise the worked sacrifice, both sides of Optimizer's strict threshold, compulsory self-hits, uniform draw slots for Random, and mirrored decisions for every deterministic agent.
