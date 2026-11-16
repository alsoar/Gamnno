# Heuristic baseline

The policy picks one complete legal turn for a known roll. It favors captures
that leave its final blot count level or lower. It uses every legal way to
play the dice, including a backward move that sets up a capture with the
next die.

## Use the agent

```python
from backgammon import starting_state
from heuristic import choose_turn, evaluate_turn, tier

state = starting_state()
turn = choose_turn(state, (2, 3))
metrics = evaluate_turn(state, turn)
print(turn.moves, tier(metrics), metrics.backward_distances)
```

The returned `Turn` carries both the ordered moves and the next immutable
state. A blocked player passes. Asking for a turn in a finished game raises
`ValueError`. The optional keyword `prefix` keeps steps already played when
a person hands a partial turn to the agent in the visualizer.

For example, give White one checker on point 3 and fourteen on 10. Put two
Brown checkers on 6 and thirteen on the bar. With a 2 and a 3, White first
moves a Brown checker from 6 to 8 using the 2, then moves White from 3 to
6 using the 3. The turn earns one capture and three forward pips. The
friendly blot count remains one.

## How turns are compared

An immediate win comes first. For other turns, the agent compares how many
times each legal sequence sends one of its own checkers to the bar. It
always considers the sequences with the fewest self-hits. A sequence with
more self-hits must satisfy both conditions.

1. Each extra self-hit is followed, on the same turn, by a distinct capture
   in the opponent's six-point home quadrant (points 0–5 for White to hit
   in, or 18–23 for Brown).
2. Compared with every alternative that costs fewer self-hits, the sequence
   gains at least one additional capture in that quadrant per extra self-hit.

For example, White can move Brown onto a white blot, then use the next die
to re-enter and hit a Brown blot on point 2. If White could already make
that quadrant capture without a self-hit, the sacrifice does not qualify.
Keeping the same number of blots takes precedence over voluntarily sending
a blot to the bar just to lower the count. Forced moves and steps already
played in a partial turn are accounted for in the comparison.

Among the eligible turns, the first applicable row wins. Within that row,
compare criteria from left to right. A *free capture* means at least one opposing checker hit with no
increase in the moving player's number of blots.

| Priority | Qualifying turn | Comparison within the row |
| --- | --- | --- |
| 0 | Bears off and wins | Fewest self-hits, then stable move order |
| 1 | Free capture with at least one hit outside the home quadrant | Most captures, most outside hits, greatest capture value, lowest exposure change, most forward pips |
| 2 | Any other free capture | Most captures, greatest capture value, lowest exposure change, most forward pips |
| 3 | Reduces exposure | Fewest final blots, most captures, most forward pips |
| 4 | Holds exposure level | Most captures, farthest starting distances, most forward pips |
| 5 | Increases exposure | Smallest exposure increase, most captures, farthest starting distances, most forward pips |

After each row's comparisons, prefer fewer self-hits. Exact ties use the
lexicographically first move sequence in the moving player's orientation.
Each move compares own checker before opposing checker, then source,
destination, and die, in ascending order. This makes a reflected position
choose the reflected turn.

### Metric conventions and ambiguities

- Count a capture at the instant a moved checker lands on a blot. An opposing
  checker moved onto one of our blots counts as a voluntary self-hit. Count
  the hit even if we re-enter on the next die. The existing bar rule allows
  only the active player to re-enter their own checkers.
- A blot is one on-board checker of the acting color. Exposure change compares
  the initial and final position. A temporary blot between dice is allowed.
- Capture location is the point where the hit occurs. White's six-point home
  quadrant is 18–23 and Brown's is 0–5. The twelve-point home half continues
  to govern bearing-off eligibility. A captured Brown checker on point
  `i` is worth `23 - i`, and a captured White checker is worth `i`. Sum these
  integer progress values over all opposing checkers hit.
- Only moving our own checker earns forward pips. An entry moves by its die
  value. An oversized bear-off earns only the remaining distance to the tray.
  For each own step, measure its source distance at the time it starts. Bar
  entry starts 25 pips away. Sort these distances from largest to smallest
  and compare the lists lexicographically.
- The maintain-exposure priority lists total progress before rear-checker
  distance, while the explicit racing fallback reverses them. A level-exposure
  capture already qualifies for priority 1 or 2. For the remaining quiet
  turns, this implementation gives the explicit fallback precedence. It can
  choose a shorter advance by a farther-back checker.
- The rules engine returns one representative per final board. Two paths to
  that board can hit at different points or use the dice in a different
  order. The agent compares those paths with the engine's move and transition
  functions. It shares final states and caches legal continuations during
  the roll. Some selected sequences therefore differ from the representative
  in `legal_turns`. They replay with `apply_turn` to the same legal state.

## Reproduce the self-play run

```sh
python3 selfplay.py --games 100 --seed 2026 --turn-limit 300
```

`play_game(agent_white, agent_brown, seed=None, max_turns=300)` returns the
winner, observed turn count, move/event totals and final state.
`simulate_games(n_games, agent_white, agent_brown, seed=2026, max_turns=300)`
returns the aggregate statistics as a JSON-ready dictionary. Pass a callable
or an object with a `choose_turn(state, dice)` method. The default seed for
both entry points is 2026. Game zero uses the supplied seed and each
successive game increments it by one. A turn is one
player's roll, including a forced pass. A win on the last permitted turn
counts as a win.

### Same-seed comparison

Both versions played the same 100 opening seeds (2026–2125) to a first
bear-off. A player turn includes the opening roll and any forced pass.

| Completed games | Previous policy | Revised policy |
| --- | ---: | ---: |
| White wins / Brown wins | 48 / 52 | 52 / 48 |
| Mean player turns to win | 727.04 | 94.86 |
| Approximate 95% confidence interval for the mean | 545.80–908.28 | 73.45–116.27 |
| Median player turns to win | 472.5 | 55.5 |
| Shortest / longest | 14 / 6,088 | 13 / 590 |

The intervals use Student’s *t* with 99 degrees of freedom across completed
game lengths. The revised policy had 69 self-hits and 2,367 opponent captures
over all 9,486 turns. On each of those 69 turns a self-hit-free alternative
existed, and the chosen self-hit enabled an additional subsequent capture
in the opponent’s home quadrant. The longest revised game ended on turn 590, so the command
above with `--turn-limit 1000` also reproduces all 100 revised wins.
[Revised per-seed results](heuristic-revised-100-games.csv) · [Previous per-seed results](heuristic-uncapped-100-games.csv)

The 300-turn command above gives both policies the same observation window.

| Through the earlier of victory or turn 300 | Previous policy | Revised policy |
| --- | ---: | ---: |
| Games finished / still playing at 300 | 41 / 59 | 94 / 6 |
| White wins / Brown wins | 22 / 19 | 47 / 47 |
| Mean blots after each observed turn for its moving player | 0.665 | 0.642 |
| Opponent captures per game started | 63.00 | 21.13 |
| Self-hits per game started | 23.39 | 0.63 |
| Self-hits per 1,000 checker moves | 45.39 | 3.25 |
| Backward steps / all checker steps | 11,322 / 51,536 | 4,542 / 19,407 |

The machine-readable reports give exact turn-length histograms and event totals.
[Revised 300-turn report](heuristic-revised-selfplay.json) · [Previous 300-turn report](heuristic-selfplay.json)
