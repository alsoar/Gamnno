# magnno engine benchmarks

On this machine, the compact engine generated **360,000 legal turn outcomes per second**. The opening position used **27 bytes when serialized** and **212 bytes as a live Python state**. The forward-priority self-play policy completed **25.4 games per second**.

## Reproduce the measurements

Run this from the source directory with NumPy installed.

```sh
python3 benchmark.py --rounds 20 --games 24 --turn-limit 300 --seed 2026
```

The figures below are independent medians from three invocations with the same inputs on macOS 26.2, ARM64, CPython 3.12.14 and NumPy 2.3.5. Timings use one process and wall-clock seconds. The default command, `python3 benchmark.py`, uses three rounds and six games per policy for a shorter check.

## Memory per opening position

| Measurement | Bytes |
| --- | ---: |
| 24 signed physical points | 24 |
| Serialized position including both bars and the active player | 27 |
| Compact state and all uniquely reachable backing storage | 212 |
| Original engine’s opening board paired with active player | 816 |
| Legacy export paired with active player, fresh count pairs | 1,824 |
| Legacy export paired with active player, with equal count pairs shared | 704 |

The compact state uses a 40-byte Python wrapper, a 112-byte NumPy array header and a 60-byte immutable bytes object. The bytes object already includes its 27-byte payload. The array and serialization key reference the same storage. Holding an additional public `state.board` view takes another array header and shares the original data.

Live sizes use `sys.getsizeof` on each distinct reachable object once, including NumPy backing storage. The calculation excludes shared interpreter integers, classes, allocator slack and process-wide memory. The original opening layout shares its empty count pair across points. Its live size depends on which cell tuples are shared, particularly across successive turns.

## Legal-turn generation

Each round used the opening setup, a contact position, a position with both colors on the bar, and a race where either color could bear off. Every position had 15 checkers per side. Both players received all 36 ordered rolls, including the six doubles. That weights doubles as often as two fair dice do. State setup and one warm-up with each of two rolls per position were outside the timer.

| Engine | Calls | Distinct outcomes returned | Seconds | Calls/s | Outcomes/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact state | 5,760 | 585,000 | 1.626 | 3,543 | 359,803 |
| Earlier tuple engine, same workload | 5,760 | 585,000 | 2.303 | 2,501 | 253,966 |

Each call builds its complete set of legal successor states, including deduplication. Outcomes are counted separately for every call. The compact implementation produced **1.42×** the throughput of the earlier engine on this workload. Individual compact runs ranged from 234,061 to 365,557 outcomes/s.

For a rollout that needs one random successor, `sample_turn` still enumerates distinct final positions but creates a public state only for the chosen outcome. On the same positions and dice, it served **4,724 sampled turns/s** (5,760 calls in 1.219 seconds). Building all successors served 3,543 calls/s.

## Self-play

Each policy played 24 games, starting from the standard opening roll, with seeds 2026 through 2049. Both sides used the same policy. The timer includes dice, move generation, policy selection and winner detection. A turn counts one player's roll and resulting action, including a pass. Capped sessions stop after 300 turns. Only actual bear-offs contribute to finished games/s.

| Policy | White wins | Brown wins | Capped | Player turns | Seconds | Player turns/s | Finished games/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward priority | 13 | 11 | 0 | 1,829 | 0.944 | 1,937 | 25.42 |
| Uniform random | 0 | 0 | 24 | 7,200 | 3.810 | 1,890 | 0.00 |

Forward priority takes an immediate winning turn when available, then favors the largest number of dice spent moving its own checkers. It breaks ties uniformly among the generated representatives. Uniform random calls `sample_turn`, giving every distinct legal successor equal probability. Its 24 games all reached the cap, so this sample gives a turn rate but no estimate of how long an uncapped random game takes.

## Rule preservation

During migration, the compact engine, archived tuple engine and an independent checker-by-checker reference model agreed on **31,471 legal successor boards across 368 position, player and roll combinations**. The comparison covered every unordered roll and both players on the four benchmark layouts, plus 100 seeded asymmetric positions and their color reflections. All returned successors switched the active player, survived serialization, and had the same winner as the original. A sample of representative move sequences was also replayed through the public API.

An additional 504 seeded checks confirmed that `sample_turn` chose the same representative as a uniform draw from `legal_turns` with the same random seed.
