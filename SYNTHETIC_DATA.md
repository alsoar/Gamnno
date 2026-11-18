# Synthetic pre-roll value data

Each primary example asks whether the player to move eventually won this particular game. The outcome reflects the policies that played the game. Different visits to the same position can have different labels. The loader builds its input from the 24 signed point counts, both bars and the active color. Opening-roll records, dice, later actions and policy choices are available in the replay archive. The opening roll determines both the opener and the dice, so the first eligible value sample is the position before rolling on turn two.

## Generate and validate

Install the repository requirements in a Python environment. The following commands use separate directories for each dataset and compute a validation report at the end. Running the pilot command again with the same arguments resumes its checkpoint.

```sh
python -m pip install -r requirements.txt
python synthetic_data.py pilot --workers 8 --output data/synthetic_v1/pilot
python synthetic_data.py audit data/synthetic_v1/pilot
```

After reviewing the pilot, start full generation explicitly.

```sh
python synthetic_data.py full --workers 8 --output data/synthetic_v1/full
```

The pilot targets 5,000 primary completions and 100 independent generalization games. The full setting targets 80,000 primary completions, about 1.2 million primary samples and 1,000 additional generalization games. Each game stops on a real bear-off or after 2,000 player turns, including passes. A win on the last turn counts. Capped attempts remain in the game catalog with a null winner and reason `turn_limit`. They produce zero primary rows. A terminal board never becomes a pre-roll input.

Set `--games`, `--heldout-games`, `--seed`, `--max-turns`, `--positions-per-game`, and `--max-attempts-per-slot` to change targets and budgets. For a small run, use a fresh output directory. To bound work per invocation, add `--attempt-budget 1000`. A budget-limited run exports and audits its available data, including any target shortfalls. Resume with the same command and the same or a higher per-slot retry ceiling. The default allows up to 64 attempts for each requested completion. The full preset archives selected trajectories. Use `--trajectories all`, `sampled` or `none` and optionally `--trajectory-rate 0.02`. In sampled mode at least one slot per scheduled major category is reserved for archiving. The pilot archives every attempt, including capped games.

For a different policy or sampling mix, pass `--config settings.json`. This example shows the three supported sections. Omitted settings use the preset defaults.

```json
{
  "plan": {
    "mixture_weights": {"baseline": 0.15, "sprinter": 0.15, "fortress": 0.10, "saboteur": 0.15, "assassin": 0.10, "blockader": 0.10, "optimizer": 0.15, "hoarder": 0.10},
    "epsilons": [0.05, 0.10, 0.20]
  },
  "sampling": {
    "positions_per_game": 15,
    "phase_weights": {"early": 0.20, "middle": 0.30, "late": 0.30, "near_terminal": 0.20},
    "min_turn_gap": 3
  },
  "trajectory": {"mode": "sampled", "probability": 0.02}
}
```

### Read the portable dataset

The published snapshot includes the Parquet partitions, game catalog, compact trajectory shards, manifest, configuration, numeric dictionaries, and reports. The writer's local SQLite checkpoint is only needed to resume generation. A downloaded snapshot can be loaded or audited directly. Choose a fresh output directory to run the generator on a downloaded copy.

```python
from pathlib import Path
import pyarrow.parquet as pq
from synthetic_store import state_from_row, value_input

root = Path("data/synthetic_v1/pilot")  # or the directory of a downloaded snapshot
for shard in sorted((root / "train").glob("part-*.parquet")):
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            state = state_from_row(row)       # engine's compact 27-byte position
            inputs = value_input(row)        # board and bars, oriented to the active player
            target = row["win_target"]       # 1 for a win, 0 for a loss
```

The fixed 24-byte `board` holds signed int8 counts, with separate unsigned bars. White is +1 and Brown is −1. In the model input the active player always advances toward increasing indices. The global position key is the pair `(game_id, position_id)`, with the position number equal to its one-based player turn. The remaining fields identify the player, absolute winner, game length, game seed, two seat policies, allocation category, mixture type, board phase, and acting strategy. For `source_policy_metadata`, the low seven bits index the pure-strategy dictionary and bit 7 marks an epsilon exploration move. These identifiers and all outcome-related fields belong to the metadata. Use `value_input` to construct the actual prediction input.

### Replay a game

The `games/` Parquet catalog has one row for every attempt. It includes seat configuration and game-mixture draws, the advancing strategy code for **every** turn, the opening and final board, and capture, movement and board-structure totals. For games with an archived trajectory, the catalog specifies the binary shard path, byte offset and byte length. One compressed record contains the ordered dice, complete actions, afterstates, pure strategy and hit counts for that game. Both the simulator and offline checker replay the actions through the common rules engine.

```python
from synthetic_game import iter_trajectory

catalog = sorted((root / "games").glob("part-*.parquet"))
for path in catalog:
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=128))
    recorded = next((r for r in batch.to_pylist() if r["trajectory_file"]), None)
    if recorded:
        with (root / recorded["trajectory_file"]).open("rb") as stream:
            stream.seek(recorded["trajectory_offset"])
            blob = stream.read(recorded["trajectory_size"])
        for ply in iter_trajectory(blob):
            print(ply.number, ply.dice, ply.moves)
        break
```

## Allocation, retries and generalization

The generator assigns completion slots in the ratio 35,000 cross-strategy, 10,000 pure self-play, 20,000 per-turn mixture, 10,000 per-game mixture and 5,000 exploratory. Largest-remainder rounding handles smaller runs. Cross-strategy slots use seven mutually exclusive subgroups with shares 15/15/15/15/20/15/5 in the order requested. Optimizer games form their own subgroup, the remaining nonrandom pairs exclude the reserved matchups, and Random's cross-strategy games always face a nonrandom policy. Exploratory games use a distinct allocation label even where the same players can also meet in cross-strategy play.

The pure self-play schedule cycles through the eight nonrandom policies. Turn-mixture seats select once per complete turn. Game-mixture seats draw independently at game start. Noisy seats sample one exploration coin for the whole turn and delegate either to the original deterministic policy or to the existing Random agent, which samples legal final boards uniformly. A tagged turn in a winning position can therefore choose another legal ending under Random.

Attempts are balanced by concrete opponent and color within each scheduled group. After an unfinished attempt, the slot advances to the next pairing in its own allocation catalog. A Sprinter–Saboteur slot always remains Sprinter–Saboteur. A general pure-self-play slot may switch to another pure self-pairing. Consequently a policy that seldom finishes can yield many attempted games and very few labeled games even with a balanced attempt schedule. Consult the matchup table for the realized training distribution.

The primary split contains 80% train, 10% validation and 10% test slots. Split assignments are fixed before sampling and stratified by scheduled allocation, including the cross-strategy subgroups. The winner can only be observed later, so the realized outcome balance is measured separately in each split. The separate held-out evaluation set changes both seed namespace and mixture distribution and includes two noisy opponents facing each other. Its default weights in registry order are 5/25/5/10/15/5/20/15/0 percent. Reserve it for the final generalization evaluation.

Every attempt has a stable UUID and a distinct game seed. Independent streams drive matchup order, dice, each seat's strategy selection, exploration coin and random action, and position/trajectory sampling. The same configuration reproduces game IDs, rolls, labels and actions with different worker counts, including after a restart. Wall-clock timings, memory use and timestamps are measurements of that run.

## Phase selection and duplicate handling

The classifier evaluates the board in both colors' physical directions. Each bar checker counts as an outsider 13 pips from its home half. The first applicable condition wins.

| Phase | Default board condition |
| --- | --- |
| Near-terminal | Either side is already eligible, or has an empty bar, at most one outsider, at most two total pips to eligibility and a leader within six pips of the tray. |
| Late | Either side has at most two outsiders. |
| Early | Both have at least six outsiders and at least 24 total pips to eligibility. |
| Middle | All remaining pre-roll positions. |

Override thresholds in `sampling.phase_thresholds` with `near_outside`, `near_distance`, `near_lead`, `late_outside`, `early_outside`, or `early_distance`. The near-terminal rule is a geometric proxy. The audit also counts positions that immediately preceded the actual winning turn.

Each game keeps bounded, seed-ranked reservoirs by phase and side to move. Distinct states are selected toward the 20/30/30/20 phase targets, balancing colors and preferring at least three plies of spacing. When a phase or a short game cannot supply its quota, the sampler can relax spacing and use a different available phase. It never manufactures another copy of a sampled state to fill a quota. The default limit is 15 per completed game irrespective of game length.

The export visits train, validation, test and held-out in that order. The first split to contain a model input owns it. Later splits omit any row with the same active-player-normalized board and bars, including mirror-image equivalents. Within-split repeats from different games are kept. Reports quantify physical and normalized duplicates, skipped rows, observed phase shares and shortages.

## Validation and storage

The coordinator writes each finished attempt and its samples in one transaction. Each slot can produce only one completed game. Configuration fingerprints prevent incompatible resumes. Each attempt is checked for legal moves, continuity and terminal outcome during simulation. The separate audit checks manifest file hashes, policy-code seeds, state validity, outcome and source provenance, actual split ownership, duplicate records and held-out isolation. It replays every retained trajectory. Use `python synthetic_data.py audit PATH --verify-seeds 20` to regenerate up to 20 attempts, including archived games, and independently compare the original agents’ selected actions. Selection covers allocation categories and wrapper types before filling the remainder.

The manifest gives observed attempts, completions, cutoff counts, partition sizes, generation and serialization rates, measured process memory and source hashes. `reports/REPORT.md` is the readable summary. `reports/report.json` includes matchups and seat policies, length percentiles, first-player outcomes, phase and label counts, board and movement statistics, and validation coverage. Value partitions use Zstandard-compressed Parquet. A compact binary trajectory is compressed independently so one game can be fetched by byte range. Files are written in shards, and repeated generation can resume from `checkpoints/run.sqlite3`.

Wait for the generation command and its audit to finish before loading, copying, or publishing an exported snapshot. Export replaces each shard as a complete file and publishes the final manifest after all replacements. An interruption during this step can leave shards from different export attempts, which the checksum audit rejects. With the local SQLite checkpoint present, rerun the same generation command with `--attempt-budget 0` to rebuild the snapshot and validate it from the saved games. This recovery retains the recorded attempts and samples.
