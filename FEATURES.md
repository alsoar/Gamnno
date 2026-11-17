# Complete-turn features

`iter_candidates(state, dice, bar_penalty=3)` yields one `Candidate` for every legal ordered turn. It retains different die orders, including those that reach the same board. Each candidate has an engine `turn`, the moving and opposing players’ position summaries before and after that turn, and a chronological record of hits. A forced pass has an empty move sequence. A finished game has no candidates.

`prefer_win=True` returns just the first canonical winning path when a win exists. Otherwise it yields every legal path. `prefix=...` limits the search to continuations of a committed sequence and keeps the metrics for the whole turn. Canonical order follows the existing enumerator, comparing moves from the acting side’s perspective by own checker before opposing checker, source, destination, then die. A sequence ending on the first bear-off stops immediately.

`evaluate_chosen(state, turn, bar_penalty=3)` explains one known-legal turn without generating alternatives. `side_features(state, player)` describes a single color. `feature_record(candidate)` produces JSON-ready features for sampled game records. The per-color fields are `own_before`, `own_after`, `opp_before` and `opp_after`. The actor is always the player at the beginning of the turn.

For individual position queries, pass `(state, player)` to `blots`, `effective_vulnerability`, `outside_home_half`, `distance_to_home_half`, `leading_checker_distance`, `occupied_points`, `made_points`, `stack_concentration`, `longest_blockade`, `opponent_forward_mobility`, or `capture_opportunities`. The mobility helper describes the opponent of the supplied player. The opportunity helper describes that player’s own threats. `exposure_change(before, after, player)` compares two positions. For several simultaneous queries, read the cached fields on the same `Side` or `Candidate`.

The named turn accessors are `captures`, `self_captures`, `outside_home_captures`, `capture_value`, `total_forward_progress` and `backward_displacement`. Pass a measured candidate to reuse its event records, optionally supplying its acting player as a second argument. When starting with a raw legal turn, pass `(initial_state, turn)`. Recovering capture events requires the position before the turn. `immediate_win(turn, player)` accepts a raw turn or a candidate and uses the final state alone.

## Coordinates, progress and exposure

For either color, relative point **0** is the first point after re-entry and **23** is adjacent to its exit. The bar is conceptually at −1 and the exit at 24. The actor’s home half spans 12–23. Its standard home quadrant spans 18–23. The opponent’s quadrant spans 0–5 in the actor’s coordinates.

| Field | Definition |
| --- | --- |
| `blots`, `bar` | Number of occupied single-checker points, and number of that color’s checkers in jail. |
| `effective_vulnerability` | Blots plus `bar_penalty × bar`. The default penalty is 3. |
| `outside_home_half` | Checkers on points 0–11 plus all own bar checkers. |
| `distance_to_home_half` | Sum of `max(12 − point, 0)` for each checker on the board, plus 13 pips for each checker on the bar. |
| `eligible` | All remaining checkers are in the home half, including the requirement for an empty own bar. |
| `leading_checker_distance` | Distance of the most advanced checker to the exit, 25 when all are on the bar, 0 after bearing off. |
| `bearoff_candidates` | When eligible, number of checkers within six pips of the exit. This measures a possible exact one-die finish. |
| `victory_distance` | `25 × outside_home_half + distance_to_home_half + leading_checker_distance`. Smaller values mean closer. This is a positional score, not a predicted turn count. |

`forward_progress` counts the actual pips advanced by the actor’s own checkers, including entry and the distance remaining on an overshoot. `backward_displacement` and `backward_moves` count the pips and steps spent pushing opponents. `backward_distances` gives the descending source distances of own forward movers from their exit, measured at the time of each step. `opponent_backward_distances` gives the corresponding descending distances of opponents from reaching their home **half** before each push. An opposing checker already home contributes zero to this latter tuple.

`exposure_change` is the final own blot count minus the initial own blot count. An intermediate blot secured later in the turn does not worsen final exposure. A self-hit is still recorded even if the actor re-enters with the next die. Policies compare vulnerability, progress and the actual number of self-hits when evaluating a sacrifice.

## Captures at impact

Each entry in `events` contains the zero-based step number, physical engine move, self-hit flag, quadrant flags, victim progress and capture bonuses. Count `captures` only when an opposing checker goes to the bar. Count `self_captures` when a backward move lands on the actor’s own blot. Bar occupancy can change again before the turn ends.

For an opposing blot, capture value is a sum in pip-like units.

| Contribution | At the instant of the hit |
| --- | --- |
| Progress | Victim’s relative point, from 0 at its entry edge to 23 adjacent to its exit. |
| +12 | The opponent was eligible to bear off. Sending this blot to the bar breaks eligibility. |
| +6 | The victim was the opponent’s only checker outside its home half, with none already on its bar. |
| +6 | The opponent was eligible and this victim was its only checker within exact one-die bearing-off range. |

For example, hitting a Brown blot on physical point 2 gives 21 progress units. If Brown was eligible and this was its only checker ready for a one-die finish, the value is 39. The sole-outside bonus and the eligibility bonus are mutually exclusive. A self-hit has zero opposing capture value. `outside_home_captures` counts hits outside the actor’s last six points. `opponent_home_captures` counts the narrower subset inside the opponent’s own last six.

## Structure and opportunity

| Field | Definition |
| --- | --- |
| `occupied_points`, `made_points` | Points containing at least one or at least two own checkers respectively. |
| `stack_concentration` | Sum of squared own stack sizes on physical points. The bar and trays contribute zero. |
| `longest_blockade` | Longest consecutive run of own made points. |
| `made_mask`, `blockade_pairs` | Actor-relative bit masks for made points and the starting indices of adjacent made pairs. Comparing before and after identifies broken or preserved structures. |
| `rear_home_stack` | On relative points 12–17, sum `(18 − point) × own_count²` over made points. Larger rear-home stacks score higher. |
| `nearby_blocks` | Made points within one to six forward pips of any enemy checker, or on an entry point when the opponent has a checker on its bar. Each blocking point counts once. |
| `stack_joins`, `large_splits` | Friendly steps landing on an existing friendly point, and friendly steps leaving a stack that previously contained at least three. These are movement history, not board totals. |

`forward_mobility` is a six-face estimate for the color being described. For each on-board checker, count die faces that lead to an empty point, a friendly stack, a capturable blot, or an allowed bear-off. Divide by six times the number of on-board checkers. When that color has a checker in jail, use only the fraction of its six unblocked re-entry points. To evaluate the opponent’s mobility, read `opp_after.forward_mobility`.

`threats` counts distinct opposing origins where a future capture appears geometrically possible. A blot counts if a friendly checker can reach it in one die, or if the blot can first be pushed backward onto an empty point within one-die reach. A two-checker point counts if a friendly checker can reach it and a backward push could split the pair without hitting a friendly checker. With a friendly checker on the bar, only opposing entry blots count. These positional estimates ignore real future rolls, mandatory die usage and other turn-wide constraints.

## Evaluation cost

The existing engine determines legal final states and full-roll obligations. The ordered-path enumerator expands alternate legal histories using the engine’s step and transition functions. A per-decision edge cache holds step annotations, so alternative routes do not replay the same capture calculation for a shared edge. A second per-decision cache computes both color summaries once per distinct final board. Candidate paths stream through the policy with at most four moves each. Cache storage is released after the decision. The standalone extractor replays at most four chosen moves for telemetry.
