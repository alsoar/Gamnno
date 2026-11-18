# Synthetic dataset audit

**The primary target remains short by 318 completed games.**

The agreed 10,000-attempt budget is exhausted. [Recorded resource budget](resource-budget.json).

The archive contains 10,000 attempts and 4,782 completed games. The value partitions contain 69,847 pre-roll positions and 67,547 distinct boards including active color. The primary allocation reached 4,682 of 5,000 completions. The separate generalization set reached 100 of 100.

Binary training examples come from games that completed within the 2,000-turn limit. The unfinished attempts retain their diagnostics and supply no binary outcome labels. When completion rates differ by policy, the training sample emphasizes positions and matchups that finish within this limit. This selection also affects the phase and strategy mix.

## Validation

The audit checked checkpoint labels, deterministic allocations, player-to-move parity, phase assignments, policy traces and the exported Parquet rows. It replayed 10,000 archived games and independently checked 70,689 sampled states against their exact turn. It also regenerated 12 games from their seeds to verify the policy choices and complete action sequences. 0 games (0 sampled states) rely on the simulator's online action and reachability checks.

Exact repeated positions within splits account for 3.29% of rows. 842 candidate rows were withheld to prevent equal board/turn-color inputs from appearing in different splits after player normalization. The exported partitions have 67,211 distinct player-normalized model inputs and no cross-split overlap.

## Sampling and labels

| Split | Positions | Win labels | Early | Middle | Late | Near terminal | On winning turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| heldout_evaluation | 1,369 | 50.0% | 11.2% | 33.6% | 38.6% | 16.5% | 5.4% |
| test | 6,596 | 49.7% | 14.7% | 35.1% | 36.0% | 14.2% | 5.4% |
| train | 55,316 | 49.9% | 17.5% | 34.6% | 34.4% | 13.6% | 5.1% |
| validation | 6,566 | 49.9% | 15.0% | 34.5% | 36.0% | 14.5% | 5.3% |

Phase targets (early, middle, late, near terminal) are 20%, 30%, 30%, 20%. The final column counts sampled states immediately before the winning turn. The near-terminal phase uses board geometry.

## Completion and game lengths

| Cohort | Attempts | Wins | Unfinished | Completion | Mean completed turns | Median completed turns | 90th percentile | Opening player win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heldout_evaluation | 145 | 100 | 45 | 69.0% | 193.1 | 44.5 | 737.4 | 62.0% |
| primary | 9,855 | 4,682 | 5,173 | 47.5% | 261.8 | 57.0 | 972.0 | 51.5% |

Opening-player rates use completed games. Each observed length includes the final turn or, for unfinished attempts, every turn through the limit.

### Primary allocation categories

| Category | Target | Attempts | Completed | Shortfall | Unfinished | Completion rate | Mean observed turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| cross_strategy | 2,188 | 5,601 | 1,870 | 318 | 3,731 | 33.4% | 1,407.9 |
| exploratory | 312 | 589 | 312 | 0 | 277 | 53.0% | 1,032.5 |
| game_mixture | 625 | 972 | 625 | 0 | 347 | 64.3% | 830.9 |
| pure_selfplay | 625 | 912 | 625 | 0 | 287 | 68.5% | 773.8 |
| turn_mixture | 1,250 | 1,781 | 1,250 | 0 | 531 | 70.2% | 878.4 |

Sprinter–Saboteur completed 15 of 2,833 attempts (0.5%). Its allocation target is 328 completed games, leaving 313 unfilled. The finished games provide all binary examples for this matchup. Its unfinished trajectories describe additional positions encountered during play.


### Pure self-play coverage

| Policy | Attempts | Completed | Completion rate | Share of attempts | Share of completed games |
|---|---:|---:|---:|---:|---:|
| assassin | 138 | 0 | 0.0% | 15.1% | 0.0% |
| baseline | 156 | 147 | 94.2% | 17.1% | 23.5% |
| blockader | 216 | 216 | 100.0% | 23.7% | 34.6% |
| fortress | 78 | 75 | 96.2% | 8.6% | 12.0% |
| hoarder | 87 | 87 | 100.0% | 9.5% | 13.9% |
| optimizer | 78 | 19 | 24.4% | 8.6% | 3.0% |
| saboteur | 78 | 0 | 0.0% | 8.6% | 0.0% |
| sprinter | 81 | 81 | 100.0% | 8.9% | 13.0% |

A self-play game appears once in this table. The generator starts with an evenly distributed policy catalog within one broad self-play completion quota. Retry slots advance through that catalog. Different completion rates can therefore skew both attempted and completed policy shares. The completed-policy shares above measure the resulting representation.

### Positions encountered before sampling

| Primary games | Pre-roll visits | Early | Middle | Late | Near terminal |
|---|---:|---:|---:|---:|---:|
| completed | 1,220,929 | 66.8% | 22.5% | 9.5% | 1.2% |
| unfinished | 10,340,827 | 97.3% | 2.4% | 0.3% | 0.0% |

These counts include every pre-roll start after the opening. Longer games contribute more visits. The per-game sampling limit bounds their contribution to the released value dataset.

### Primary agent seats

| Policy | Appearances | Completed | Win rate on completions | Mean observed turns |
|---|---:|---:|---:|---:|
| assassin | 1,009 | 480 | 99.8% | 1,183.5 |
| baseline | 772 | 636 | 62.1% | 614.4 |
| blockader | 1,575 | 1,342 | 25.9% | 470.7 |
| fortress | 887 | 833 | 15.2% | 392.7 |
| game_mixture | 1,320 | 836 | 47.0% | 848.5 |
| hoarder | 1,233 | 867 | 40.0% | 649.8 |
| noisy:assassin:0.05 | 12 | 8 | 100.0% | 925.8 |
| noisy:assassin:0.1 | 6 | 0 | — | 2,000.0 |
| noisy:assassin:0.2 | 10 | 10 | 90.0% | 284.3 |
| noisy:baseline:0.05 | 4 | 4 | 25.0% | 39.2 |
| noisy:baseline:0.1 | 10 | 0 | — | 2,000.0 |
| noisy:baseline:0.2 | 4 | 4 | 75.0% | 119.2 |
| noisy:blockader:0.05 | 5 | 5 | 0.0% | 71.2 |
| noisy:blockader:0.1 | 4 | 4 | 0.0% | 37.5 |
| noisy:blockader:0.2 | 4 | 4 | 75.0% | 86.2 |
| noisy:fortress:0.05 | 4 | 4 | 0.0% | 21.5 |
| noisy:fortress:0.1 | 10 | 5 | 0.0% | 1,558.4 |
| noisy:fortress:0.2 | 5 | 5 | 0.0% | 109.6 |
| noisy:hoarder:0.05 | 14 | 14 | 100.0% | 29.7 |
| noisy:hoarder:0.1 | 5 | 5 | 0.0% | 23.0 |
| noisy:hoarder:0.2 | 4 | 4 | 50.0% | 43.2 |
| noisy:optimizer:0.05 | 10 | 10 | 100.0% | 29.5 |
| noisy:optimizer:0.1 | 8 | 0 | — | 2,000.0 |
| noisy:optimizer:0.2 | 4 | 2 | 50.0% | 1,416.0 |
| noisy:saboteur:0.05 | 15 | 0 | — | 2,000.0 |
| noisy:saboteur:0.1 | 8 | 0 | — | 2,000.0 |
| noisy:saboteur:0.2 | 4 | 4 | 100.0% | 715.5 |
| noisy:sprinter:0.05 | 28 | 28 | 0.0% | 296.4 |
| noisy:sprinter:0.1 | 5 | 5 | 0.0% | 86.4 |
| noisy:sprinter:0.2 | 4 | 4 | 25.0% | 15.8 |
| optimizer | 998 | 625 | 86.1% | 842.6 |
| random | 708 | 293 | 0.0% | 1,236.6 |
| saboteur | 4,297 | 366 | 99.5% | 1,888.8 |
| sprinter | 3,763 | 924 | 51.9% | 1,522.6 |
| turn_mixture | 2,961 | 2,033 | 56.9% | 931.4 |

A self-play game contributes two agent appearances. Comparisons reflect the assigned opponents. Counts for each White/Brown matchup are available in the JSON report.

## Primary game behavior

| Metric | Observed value |
|---|---:|
| backward pips per attempt | 1,793.854 |
| backward step fraction | 0.214 |
| captures per attempt | 496.964 |
| forward pips per attempt | 7,752.079 |
| mean bar | 0.010 |
| mean blockade | 2.257 |
| mean blots | 2.503 |
| mean concentration | 68.018 |
| self captures per attempt | 5.002 |
| side eligible after ply fraction | 0.001 |
| total actor turns | 11,571,611.000 |
| total checker steps | 26,896,945.000 |

Per-attempt totals include unfinished games. Blots, bar, concentration and blockade use the active player's position after its turn. Eligibility uses both players after each turn.

## Performance

| Measurement | Value |
|---|---:|
| attempts per second | 0.330 |
| completed per second | 0.158 |
| current dataset bytes before report | 368,421,756.000 |
| generation seconds | 30,345.520 |
| mean worker simulation seconds per attempt | 24.138 |
| parent peak mib | 101.734 |
| peak memory mib | 101.734 |
| positions per second | 2.302 |
| serialization bytes | 164,728,403.000 |
| serialization bytes per second | 44,952,760.287 |
| serialization seconds | 3.664 |
| simulation worker seconds | 241,380.296 |
| storage bytes | 164,728,403.000 |
| worker peak mib | 47.750 |

Generation seconds measure recorded simulation and coordinator time. Export, audit, shutdown, and time lost before an interrupted checkpoint are excluded. Worker seconds sum the individual simulation durations. Mean worker simulation time is per attempted game. Serialization speed uses bytes written divided by measured serialization time. Storage bytes cover portable data and configuration, excluding the manifest, reports and checkpoint. Current dataset bytes measure all files present immediately before this audit writes its reports. A dash means a measurement was unavailable. Largest measured process high-water mark among the coordinator and workers. Combined concurrent RSS was unavailable.
Game outcomes, attempts and sample counts are exact archive counts. Timing and memory describe the measured execution environment and may vary on another machine.

## Observations

- Among color-specific matchups with at least five attempts, random (White) vs random (Brown) had the lowest completion rate at 0/180.
- noisy:assassin:0.1 had the longest mean observed game length among primary policy seats (2,000.0 turns across 6 appearances).
- Among exploratory seats with at least five completed appearances, random had the lowest observed win rate (0/293). Opponents and sample sizes differ across policies.
- The largest primary phase shortfall was near_terminal. Its sampled share was 13.7% against a 20.0% target.
- From the side to move, 49.9% of primary positions have a winning label.
- The most frequently sampled exact board and active color appeared 59 times.

## Recommendations before full generation

- 5,173 of 9,855 primary attempts reached the turn limit. Review completion rates by matchup before the 80,000-game run. A larger completed-game target continues selecting games that finish within the cutoff, so increasing volume alone cannot restore the omitted trajectory distribution.
- In pure self-play, assassin completed 0/138 and saboteur completed 0/78. Review the allocation and evaluate explicit completed-game targets for each policy under a fixed attempt or time budget. The current rotating catalog permits retries to change policies within a shared category quota, so raising that quota alone does not ensure balanced completed policies. Record revised quotas and unmet targets in a separately versioned pilot before scaling.
- Near-terminal positions account for 13.7% of primary samples against the 20.0% target. Review the board-feature thresholds and the distinct positions available in completed trajectories, then test phase-quota changes on a new pilot. Keep the shortage explicit when the trajectories provide too few qualifying positions.
