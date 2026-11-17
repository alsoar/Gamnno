# Simulation, tournaments and training samples

Run these commands from the repository directory after installing its requirements. One player turn means one roll and the resulting complete action, including a forced pass. The opening turn and winning turn both count.

## Run games

```sh
# One head-to-head matchup, 10 games in each color
python simulation.py matchup baseline assassin --games 20 --seed 2026 --max-turns 2000 --out matchup.json

# All nine agents, 45 unordered pairings, 900 games
python simulation.py tournament --games 20 --seed 2026 --max-turns 2000 --workers 6 --out tournament.json

# A smaller installation check, two games of Sprinter self-play
python simulation.py tournament --agents sprinter --games 2 --max-turns 2000 --out smoke.json
```

The command prints a summary when `--out` is supplied. Omit `--out` for JSON on standard output. `--games` means total games per pairing. An even number balances colors between two distinct agents. The simulator's default limit is **2,000 player turns per game**. A win on turn 2,000 counts. A game still in progress after that turn is unfinished.

```python
from agents import AGENT_NAMES, create_agent
from simulation import play_game, run_matchup, run_round_robin

adjusted = create_agent("fortress", {"bar_penalty": 4})
one = play_game(adjusted, "saboteur", seed=2095, max_turns=2000)
print(one.winner, one.turns, one.white_stats["captures"])

pair = run_matchup(adjusted, "sprinter", 8, seed=100)
report = run_round_robin(AGENT_NAMES, 20, seed=2026, max_turns=2000)
print(report["matrix"]["sprinter"]["fortress"])
print(report["profiles"]["hoarder"]["mean_concentration"])
print(report["diversity"]["requested_comparisons"])
```

The simulator accepts registry IDs or named instances. A custom agent supplies `.name` and `.choose_turn(state, dice, rng=None)`. The nine built-in agents return their already-scored candidate directly to the simulator. Custom agents without that extra method incur a feature pass over their selected turn. Give separately configured competitors distinct names when including both in the same round robin. Multiprocessing requires importable, picklable agents and the usual `if __name__ == "__main__"` entry guard for Python scripts.

## Dice pairing and reproducibility

For two distinct agents, games 0 and 1 use the same seed and reverse colors. Games 2 and 3 share the next seed, and so on. The opening color and subsequent pair of dice on every numbered turn match through the shorter game. The agents can choose different moves or finish on different turns. For an odd number of games, the first named agent plays White in the final unpaired game.

Self-play uses a different seed for every game so a deterministic strategy does not repeat identical trajectories solely to reverse names. The same seed schedule is used across all matchups. Each game has a separate dice generator, a separate action generator for each agent, and, when requested, a separate sample generator. Random choices therefore do not consume future dice. Every worker plays whole games, and the parent merges results and writes optional samples in schedule order. At most twice the worker count is queued.

## Read the reports

- **Outcomes.** The matchup list counts each actual game once. `a` and `b` are the two named participants in the listed order. The matrix gives the row agent's perspective. `win_fraction_all` divides wins by starts. `win_fraction_completed` divides wins by wins plus losses. An unfinished game contributes to neither wins nor losses. A zero denominator produces JSON `null`. Multiply a fraction by 100 to display a percentage.
- **Self-play.** A diagonal cell pools the two seats. For example, two self-games, one finished and one capped, yield four appearances, one win, one loss and two unfinished appearances. The diagonal rates are 25% of starts and 50% of completed seat appearances. The cell's `games` still reads 2. Per-agent profiles use the same seat convention.
- **Lengths.** `all_observed`, `completed` and `unfinished_observed` each contain count, mean, median and an exact length histogram. A capped game contributes its observed length to the appropriate histograms. `average_game_length` in a profile is the mean observed length over that agent's appearances.
- **Capture and movement.** Captures and self-captures count events, including intermediate hits. Per-game values divide by all starts for that matchup, or all seat appearances for that agent. Forward/backward step fractions divide by the individual checker movements. Pips are also reported per turn and per game. A forced pass adds a turn and zero checker movements.
- **Position.** Mean blots, bar occupancy, effective vulnerability, occupied and made points, concentration and longest blockade sample the acting agent's position after its turn. `mean_victory_gain` is the average reduction in its shared `victory_distance` on its own turns. The final distance is also reported. Eligibility samples both players after every turn, including the terminal position. `eligible_fraction` is the share of those observations eligible, and `eligible_plies_per_game` counts eligible observations per start.
- **Action frequencies.** Pass, forward-only, backward-only and mixed partition an agent's turns. Additional flags can overlap. They count turns with at least one capture, self-capture, bar entry, bear-off or stack join, and turns that reduce vulnerability, extend the longest blockade, increase concentration or threats, reduce distance to the home half, or break the opponent's eligibility. Both counts and shares of acting turns appear in the report.
- **Comparison.** Each pair is compared on capture-turn frequency, backward-step frequency, blots, bar occupancy, concentration, occupied points, blockade length, own-turn victory progress, eligibility and observed game length. For each measure, the absolute difference is divided by its range across the tournament's participants. The reported distance is the mean across available, varying measures. Exact equality across constant measures produces zero. A participant with no acting turns has no comparison score. Distances at or below `--near-threshold` (default 0.12) are flagged. The report includes all pairs, the three largest differences per pair, and the five specified strategic comparisons. These are descriptive similarities for this roster, seed schedule and cap. Long games contribute more observations to turn-weighted statistics.

## Record compact training examples

```sh
python simulation.py matchup optimizer saboteur --games 10 --seed 5100 --max-turns 2000 \
  --positions samples.sqlite --sample-every 5 --out sampled-match.json
```

Without `--positions`, the simulation never encodes training records. With logging enabled, `--sample-every N` samples each decision independently with probability 1/N. Use 1 to record every turn. Sampling cannot affect play. Each episode and its positions are appended in a single SQLite transaction, including games that sampled zero positions. Repeated runs receive distinct episode numbers.

Each sampled decision contains the existing 29–37 byte game record with its **27-byte pre-move state** (24 signed point counts, both bars and active color), dice and complete chosen move sequence. Strategic features and hit events are compressed separately. Episode metadata identifies the two agents, seed, schedule game identifier, result and duration. The streaming reader attaches the eventual winner to every example. Unfinished games yield `None` and have no outcome target.

```python
from training_data import iter_positions

for sample in iter_positions("samples.sqlite"):
    print(sample.episode, sample.game_id, sample.turn_number, sample.acting_agent)
    print(sample.example.dice, sample.example.action, sample.example.value_target)
    print(sample.features["self_captures"], sample.features["own_after"]["blots"])
    break
```

To build a corpus with a fixed opponent distribution, call `run_matchup` with the desired number of starts for each opponent, write to the same log and choose explicit seed ranges. A mix weighted by starts can yield a different mix of saved positions because strategies produce games of different lengths. Actor and opponent labels remain available for downstream sampling.

## Compare choices on identical positions

```sh
python simulation.py agreement results/sample-positions.sqlite --limit 100 --seed 2026 --out results/decision-agreement.json
```

Each of the five strategy pairs plays the same 100 positions using default configurations. The sample gives roughly equal weight to the nine agents that originally encountered the positions. The report counts agreement on the full move sequence and on the final board, with the number of compared positions as the denominator.

1. Scan episodes and turns in order. A decision key consists of the 27-byte state (including active color) followed by both dice in ascending order, one byte per die. Keep the first occurrence across the entire log and assign it to the agent that originally faced it. A disk-backed temporary index holds these compact keys.
2. Within each source agent, rank keys by the 8-byte BLAKE2b digest of the ASCII selection seed, a slash, and the key. Break hash ties by the key itself. Shuffle the alphabetical list of source agents with Python's `Random(seed)`.
3. Cycle through that order, taking one position per source agent per round. Skip rolls with fewer than two distinct legal final boards and continue down that same agent's ranking. If an agent has no more eligible positions, redistribute its slots among the remaining agents. The report includes a SHA-256 fingerprint of the selected keys in order.

With seed 2026, the bundled sample selected 12 positions originally played by Saboteur and 11 by each other agent. The agreement rates describe decisions on these sampled game positions. Games and opponents still vary within each source group.
