# tempo-support

Real-time transient classification for [ZTF](https://www.ztf.caltech.edu/) alerts using **TEMPO**, an evidential deep-learning model that classifies optical light curves into a hierarchical transient/variable taxonomy.

The package consumes ZTF alerts from a [BOOM](https://github.com/boom-astro) Kafka stream, runs TEMPO inference on each object's photometric history, records the result, and posts a summary (with a sunburst plot) to Slack. The trained model, its calibration/post-processing report, and the runtime cache all ship **inside** the package — no external model directory is required to run inference.

## What it does

For each ZTF alert that passes the configured Fritz filter:

1. **Fetch history** — looks up the object's `prv_candidates` (previous detections) from BOOM's MongoDB.
2. **Convert photometry** — turns ZTF `magpsf`/`sigmapsf` rows into a Fritz-style flux dataframe (microjansky, zeropoint 23.9), keeping only public program IDs (`1`, `2`) and positive-difference detections.
3. **Classify** — runs TEMPO over a configurable light-curve window (default: first 100 days), producing per-class probabilities, calibrated uncertainties, a hierarchical decision (with abstention), and an out-of-distribution flag.
4. **Annotate Fritz** — posts the leaf-class probabilities back to the source on [Fritz](https://fritz.science) as an annotation (origin `tempo`), creating a new annotation or updating the existing one for that origin.
5. **Record & report** — appends a row to a results CSV, fetches existing Fritz classifications for comparison, and posts a sunburst image + probability table to Slack (only when a Fritz classification already exists).

The model uses a 5-class hierarchical taxonomy (`transient_variable_5c`) built on an evidential transformer with Time2Vec time encoding, temperature scaling, uncertainty fusion, and selective-prediction gating.

## Installation

Requires Python >= 3.11. Dependencies are managed with [Poetry](https://python-poetry.org/).

```bash
poetry install
```

Key dependencies: `torch` (CPU is fine — inference forces CPU), `confluent-kafka`, `fastavro`, `pymongo`, `requests`, `pandas`, `numpy`, `plotly`/`kaleido` (for sunburst rendering).

## Configuration

Runtime configuration is read from environment variables, loaded from `~/.env` via `python-dotenv`.

| Variable | Purpose | Required |
| --- | --- | --- |
| `BOOM_DATABASE__USERNAME` / `BOOM_DATABASE__PASSWORD` | MongoDB auth for the `boom` database (falls back to unauthenticated `localhost:27017`) | No |
| `FRITZ_TOKEN` | Fritz API token — fetches existing classifications, resolves source URLs, and posts TEMPO annotations | No (degrades gracefully; annotations are skipped without it) |
| `SLACK_TEMPO_BOT_TOKEN` | Slack bot token for file uploads | For Slack posts |
| `SLACK_TEMPO_CHANNEL_ID` | Target Slack channel ID | For Slack posts |
| `TEMPO_BUNDLE_DIR` | Override the model bundle location (defaults to the bundled `model_bundle/`) | No |
| `CUDA_VISIBLE_DEVICES` | Forced to empty (CPU-only) by the consumer | — |

The consumer also expects:

- A **Kafka** broker at `localhost:9092` with the `ZTF_alerts_results` topic.
- A **MongoDB** instance with the `boom.ZTF_alerts_aux` collection.

## Usage

### Run the live consumer

```bash
poetry run python -m tempo_support.alerts_consumer_ztf
```

This subscribes to the Kafka topic, processes alerts that pass the `superphot_ztf` filter, annotates each source on Fritz, writes results to `results/tempo_ztf_results.csv`, logs to `tempo_ztf.log`, and posts to Slack. It commits offsets manually after each message and shuts down cleanly on `Ctrl-C`.

> **Fritz annotation scope:** annotations are posted to the **UMN TEMPO** group (`FRITZ_GROUP_IDS = [1973]`) under origin `tempo`. Adjust `FRITZ_GROUP_IDS` at the top of [alerts_consumer_ztf.py](src/tempo_support/alerts_consumer_ztf.py) to change visibility.

### Run inference programmatically

```python
from tempo_support.tempo_boom_ztf import run_tempo, leaf_class_probs

result = run_tempo(ztf_id="ZTF...", prv_candidates=prv_candidates)
if result is not None:
    print(leaf_class_probs(result))  # {class_name: probability}
```

### Smoke test on a single alert

A sample alert is included under `data/alert.json`:

```bash
poetry run python tests/test_one_alert.py
```

This prints the converted photometry, class probabilities (± calibrated std), the hierarchical decision, uncertainty gate values, and writes a sunburst plot to `data/tempo_sunburst.png`.

## Project layout

```
src/tempo_support/
├── alerts_consumer_ztf.py   # Kafka consumer → Mongo lookup → TEMPO → Fritz annotation + CSV + Slack
├── tempo_boom_ztf.py        # ZTF prv_candidates → photometry → run_tempo()
├── slack_post.py            # format_message / post_to_slack helpers
├── plot_tempo.py            # plotly sunburst of the taxonomy + probabilities
├── _tempo_helpers.py        # bundle loading, inference, post-processing, plots
├── photometry_edl/          # vendored evidential model package (transformer, losses, calibration, …)
└── model_bundle/            # trained weights, calibration report, runtime cache
data/                        # sample alert + example sunburst
tests/test_one_alert.py      # single-alert smoke test
```

## How results are structured

`run_tempo()` returns a dict whose key fields include:

- `leaf` — final per-class probabilities (`probs_final`), calibrated stds, the thresholded prediction, uncertainty scores, and an optional `ood` block.
- `decision` — the hierarchical fallback label/level and whether the model abstains completely.
- `hierarchy` — per-level predictions walking up the taxonomy.
- `n_events`, `final_horizon_days` — light-curve summary metadata.

Each processed alert is appended as one row to `results/tempo_ztf_results.csv`, including final probabilities and calibrated stds per class, the decision, the selected uncertainty gate, OOD flags, and any matching Fritz classifications.
