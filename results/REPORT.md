# magnno tournament results

**632 of 900 games finished by turn 2,000.** By turn 300, 535 had finished. Extending the cap resolved another 97 games on the same dice streams. The remaining 268 (29.8%) reached the limit with no winner.

The nine policies played all 45 unordered pairings, including self-play. Each pairing had 20 starts. Different agents played the same ten dice streams (seeds 2026–2035) in both colors. Self-play used 20 distinct seeds (2026–2045). Completed games averaged 168.6 player turns, with a median of 48.5. Counting every capped game at 2,000, the observed mean was 713.9.

## Results by agent

Each agent occupied a seat 200 times. A completed self-play game contributes one win and one loss to that agent. A capped self-play contributes two unfinished seat appearances.

| Agent | Wins | Losses | Unfinished | Wins / all starts | Wins / finished |
| --- | --- | --- | --- | --- | --- |
| Optimizer | 109 | 14 | 77 | 54.5% | 88.6% |
| Assassin | 94 | 1 | 105 | 47.0% | 98.9% |
| Sprinter | 92 | 89 | 19 | 46.0% | 50.8% |
| Hoarder | 92 | 89 | 19 | 46.0% | 50.8% |
| Blockader | 87 | 104 | 9 | 43.5% | 45.5% |
| Baseline | 85 | 61 | 54 | 42.5% | 58.2% |
| Fortress | 45 | 139 | 16 | 22.5% | 24.5% |
| Saboteur | 26 | 0 | 174 | 13.0% | 100.0% |
| Random | 2 | 135 | 63 | 1.0% | 1.5% |

Optimizer recorded the most wins (109 across 200 seats). Saboteur reached the turn limit in 174 of its 200 seat appearances.

## Matchup matrix

Read across a row. Each cell gives the row agent’s win percentage across all starts, followed by its win percentage among finished games. A dash means no game finished.

| Agent | Baseline | Sprinter | Fortress | Saboteur | Assassin | Blockader | Optimizer | Hoarder | Random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 45.0% / 50.0% | 15.0% / 15.0% | 100.0% / 100.0% | 0.0% / — | 0.0% / 0.0% | 95.0% / 95.0% | 10.0% / 22.2% | 15.0% / 15.0% | 100.0% / 100.0% |
| Sprinter | 85.0% / 85.0% | 50.0% / 50.0% | 100.0% / 100.0% | 0.0% / 0.0% | 5.0% / 5.0% | 10.0% / 10.0% | 5.0% / 5.0% | 55.0% / 55.0% | 100.0% / 100.0% |
| Fortress | 0.0% / 0.0% | 0.0% / 0.0% | 47.5% / 50.0% | 0.0% / 0.0% | 0.0% / 0.0% | 40.0% / 40.0% | 0.0% / 0.0% | 0.0% / 0.0% | 90.0% / 90.0% |
| Saboteur | 0.0% / — | 5.0% / 100.0% | 50.0% / 100.0% | 0.0% / — | 0.0% / — | 65.0% / 100.0% | 0.0% / — | 5.0% / 100.0% | 5.0% / 100.0% |
| Assassin | 5.0% / 100.0% | 95.0% / 95.0% | 80.0% / 100.0% | 0.0% / — | 0.0% / — | 100.0% / 100.0% | 0.0% / — | 100.0% / 100.0% | 90.0% / 100.0% |
| Blockader | 5.0% / 5.0% | 90.0% / 90.0% | 60.0% / 60.0% | 0.0% / 0.0% | 0.0% / 0.0% | 50.0% / 50.0% | 15.0% / 15.0% | 75.0% / 75.0% | 90.0% / 100.0% |
| Optimizer | 35.0% / 77.8% | 95.0% / 95.0% | 100.0% / 100.0% | 0.0% / — | 0.0% / — | 85.0% / 85.0% | 17.5% / 50.0% | 95.0% / 95.0% | 100.0% / 100.0% |
| Hoarder | 85.0% / 85.0% | 45.0% / 45.0% | 100.0% / 100.0% | 0.0% / 0.0% | 0.0% / 0.0% | 25.0% / 25.0% | 5.0% / 5.0% | 50.0% / 50.0% | 100.0% / 100.0% |
| Random | 0.0% / 0.0% | 0.0% / 0.0% | 10.0% / 10.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / — |

## What the policies played

Capture frequency is the fraction of acting turns with an opponent capture. Backward frequency is the fraction of individual checker moves spent moving an opponent. Board quantities are averages after the agent’s own turns. Eligibility is sampled after each player’s turn.

| Agent | Capture turns | Backward moves | Blots | On bar | Concentration | Occupied | Longest blockade | Eligible | Victory progress / turn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 38.4% | 10.5% | 1.00 | 0.00 | 64.56 | 5.06 | 2.19 | 0.2% | 13.59 |
| Sprinter | 28.8% | 0.3% | 1.86 | 0.04 | 87.00 | 4.60 | 2.05 | 0.7% | 19.88 |
| Fortress | 10.4% | 27.7% | 0.25 | 0.01 | 54.16 | 5.42 | 2.77 | 0.1% | 7.70 |
| Saboteur | 34.5% | 45.8% | 3.03 | 0.00 | 69.89 | 6.13 | 2.12 | 0.0% | 8.78 |
| Assassin | 49.8% | 12.0% | 1.78 | 0.00 | 76.22 | 5.13 | 2.31 | 0.1% | 16.30 |
| Blockader | 14.8% | 10.0% | 1.11 | 0.03 | 50.38 | 6.00 | 3.36 | 0.5% | 12.03 |
| Optimizer | 47.6% | 3.9% | 1.48 | 0.01 | 65.70 | 5.67 | 3.39 | 0.2% | 19.82 |
| Hoarder | 29.2% | 0.5% | 1.39 | 0.05 | 120.77 | 3.59 | 1.36 | 0.7% | 18.83 |
| Random | 14.3% | 43.3% | 2.68 | 0.05 | 88.51 | 5.52 | 1.63 | 0.0% | 7.96 |

| Agent | Mean observed game length (turns) | Forward moves | Captures / game | Self-captures / game | Own forward pips / turn | Opponent backward pips / turn |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 659.1 | 89.5% | 135.36 | 0.60 | 7.19 | 0.94 |
| Sprinter | 253.7 | 99.7% | 39.22 | 0.00 | 7.97 | 0.02 |
| Fortress | 435.6 | 72.3% | 23.73 | 0.01 | 5.60 | 2.51 |
| Saboteur | 1827.3 | 54.2% | 337.52 | 3.29 | 4.91 | 3.24 |
| Assassin | 1217.8 | 88.0% | 331.33 | 0.54 | 7.29 | 0.85 |
| Blockader | 214.5 | 90.0% | 17.75 | 0.06 | 7.05 | 0.92 |
| Optimizer | 828.3 | 96.1% | 212.90 | 0.01 | 7.87 | 0.25 |
| Hoarder | 237.1 | 99.5% | 38.03 | 0.00 | 7.91 | 0.03 |
| Random | 751.8 | 56.7% | 56.98 | 13.94 | 4.77 | 3.31 |

Full action counts, self-hit rates, movement distances and game-length histograms are available in the [tournament JSON](round-robin.json).

## Comparing strategies

The profile distance is the mean gap across ten behavior measures, each scaled to the observed range across agents. Distances at or below 0.12 are flagged. The same-position check uses 100 distinct state-and-roll pairs with more than one legal ending. Each source agent supplied 11 positions, with one extra from Saboteur. The two agents in each comparison received identical positions and dice.

| Strategies | Profile distance | Same board on shared rolls | Largest scaled difference |
| --- | --- | --- | --- |
| Fortress / Hoarder | 0.656 | 13.0% | Concentration (54.16 vs 120.77) |
| Baseline / Assassin | 0.161 | 51.0% | Observed length (659.13 vs 1217.83) |
| Fortress / Blockader | 0.287 | 25.0% | Bar count (0.01 vs 0.03) |
| Sprinter / Optimizer | 0.372 | 29.0% | Bar count (0.04 vs 0.01) |
| Saboteur / Assassin | 0.327 | 34.0% | Backward steps (45.8% vs 12.0%) |

The [shared-position results](decision-agreement.json) include the selection method, source agents and exact move-sequence agreement.

No pair fell within the profile threshold. Turn-weighted profiles give longer games more influence. The paired choices use identical inputs for both strategies.

## Training sample

The [compact SQLite sample](sample-positions.sqlite) contains 2,917 selected positions from 90 separately seeded games (61,022 observed turns, 64 games finished). The starting seed was 7401, with two starts per pairing and a 2,000-turn cap. Each player turn had a 1-in-20 chance of being sampled. 379 positions have a final winner. 2,538 come from unfinished games and carry a null outcome. Each example contains the 27-byte starting state, roll, acting policy, full move sequence and measured features. The [simulation guide](../SIMULATION.md) shows how to read the samples and reproduce the comparison.

## Performance

On macOS-26.2-arm64-arm-64bit with Python 3.9.6, a single simulation process played 90 starts through a win or 2,000 turns. It completed 62 and processed 67,724 player turns in 1246.2 seconds. That was 54.3 decisions/s, 0.07 starts/s and 0.05 finished games/s. The mean wall time was 13.85 seconds per start. Completed games averaged 189.1 turns. Peak process memory rose from 29.2 to 31.2 MiB over the batch. Position logging was disabled. The tournament was running in other processes on the same machine during this timing.

The decision benchmark repeated 64 position/roll cases drawn from the opening, contact, bar and race layouts, with both colors and both doubles and mixed rolls. Across this workload legal endpoint generation took 3.62 ms per roll, all ordered paths took 11.39 ms, and generating paths with their features took 41.75 ms. These are inclusive timings, measured separately. Ranking below uses already computed candidates.

| Agent | Mean decision (ms) | Decisions/s | Rank only (ms) |
| --- | --- | --- | --- |
| Baseline | 33.32 | 30.0 | 3.66 |
| Sprinter | 33.47 | 29.9 | 3.68 |
| Fortress | 34.46 | 29.0 | 4.03 |
| Saboteur | 34.20 | 29.2 | 4.62 |
| Assassin | 33.11 | 30.2 | 3.20 |
| Blockader | 34.78 | 28.7 | 3.94 |
| Optimizer | 36.60 | 27.3 | 5.69 |
| Hoarder | 33.77 | 29.6 | 4.04 |
| Random | 2.43 | 410.8 | — |

Doubles generated 3985 ordered paths per case versus 92 for mixed rolls. Generating and measuring those paths took 78.59 ms and 4.91 ms respectively. The [timings](benchmarks.json) include median latency and the complete workload parameters.

A separate CPU profile of the same four board layouts attributed 85% of total decision time on doubles to path generation and feature extraction. Within the complete decision, ordered enumeration used 38%, summarizing both sides of each ending used 23%, and aggregating step events used 17%. Endpoint summaries are shared across paths to the same board. The simulation reuses the selected features for statistics and serializes a position only when it is sampled.
