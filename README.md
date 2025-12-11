# Neptune Exporter

Neptune Exporter is a CLI tool to move Neptune experiments (version `2.x` or `3.x`) to disk as parquet and files, with an option to load them into MLflow, Weights & Biases, or ZenML.

## What it does

- Streams runs from Neptune to local storage. Artifacts are downloaded alongside the parquet.
- Skips runs that were already exported (presence of `part_0.parquet`), making exports resumable.
- Loads parquet data into MLflow, W&B, or ZenML while preserving run structure (forks, steps, attributes) as closely as possible.
- Prints a human-readable summary of what is on disk.

## Requirements

- Python `3.13`, managed via [uv](https://github.com/astral-sh/uv).
- Neptune credentials:
  - [API token](https://docs.neptune.ai/api_token/), set with the `NEPTUNE_API_TOKEN` environment variable or the `--api-token` option.
  - [Project path](https://docs.neptune.ai/projects), set with the `NEPTUNE_PROJECT` environment variable or the `--project-ids` option.
- Target credentials when loading:
  - MLflow tracking URI, set with `MLFLOW_TRACKING_URI` or `--mlflow-tracking-uri`.
  - W&B entity and API key, set with `WANDB_ENTITY`/`--wandb-entity` and `WANDB_API_KEY`/`--wandb-api-key`.
  - ZenML server connection via `zenml login` (see [ZenML docs](https://docs.zenml.io/deploying-zenml/connecting-to-zenml/connect-in-with-your-user-interactive)).

## Installation

> [!IMPORTANT]
> This project is not published on PyPI. Clone the Git repository and run it directly with `uv`.

Install dependencies in the repo:

```bash
uv sync
```

To use the ZenML loader, install with the optional extra:

```bash
uv sync --extra zenml
```

Run the CLI:

```bash
uv run neptune-exporter --help
```

## Quickstart

### 1. Export Neptune data to disk (core):

  ```bash
  uv run neptune-exporter export \
    -p "my-org/my-project" \
    --exporter neptune3 \
    --data-path ./exports/data \
    --files-path ./exports/files
  ```

Options:

| Option                                   | Description                                                                                                                                                                                                                                                                                   |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--exporter` (required)                   | `neptune3` or `neptune2`. Use the version corresponding to your workspace. For help, see the [migration overview](https://docs.neptune.ai/neptune_exporter).                                                                                                                                                                                                                  |
| `-r`/`--runs`                              | Neptune run ID filter, regex supported.<ul><li>In Neptune `3.x` the run ID is user-chosen or auto-generated, stored in `sys/custom_run_id`.</li><li>In Neptune `2.x` it's the auto-generated `sys/id` (e.g. `SAN-1`).</li></ul>                                                                                                     |
| `-a`/`--attributes`                        | One value is treated as a regex. Multiple values are treated as exact attribute names.                                                                                                                                                                                                                                   |
| `-c`/`--classes` and `--exclude`           | Include or exclude certain data types. Arguments: `parameters`, `metrics`, `series`, or `files`.                                                                                                                                                                                                                                   |
| `--include-archived-runs`                | Include archived/trashed runs.                                                                                                                                                                                                                                                                |
| `--api-token`                            | Pass the token explicitly instead of using the `NEPTUNE_API_TOKEN` environment variable.                                                                                                                                                                                                      |
| `--no-progress`, `-v`/`--verbose`, `--log-file` | Progress and logging controls for the CLI.                                                                                                                                                                                                                                                    |

#### Export examples

Export everything from a project:  
```bash
uv run neptune-exporter export -p "workspace/proj" --exporter neptune3
```

Export only parameters and metrics from runs matching a pattern:
```bash
uv run neptune-exporter export -p "workspace/proj" --exporter neptune3 -r "RUN-.*" -c parameters -c metrics
```

Export specific attributes by pattern:
```bash
uv run neptune-exporter export -p "workspace/proj" --exporter neptune3 -a "metrics/accuracy" -a "metrics/loss" -a "config/.*"
```

Export with Neptune `2.x` client, splitting data and files to different locations:
```bash
uv run neptune-exporter export -p "workspace/proj" --exporter neptune2 --data-path /mnt/fast/exports/data --files-path /mnt/cold/exports/files
```

### 2. Inspect what was exported:

  ```bash
  uv run neptune-exporter summary --data-path ./exports/data
  ```

### 3. (optional) Load into a target:

  ```bash
  # MLflow
  uv run neptune-exporter load \
    --loader mlflow \
    --mlflow-tracking-uri "http://localhost:5000" \
    --data-path ./exports/data \
    --files-path ./exports/files

  # W&B
  uv run neptune-exporter load \
    --loader wandb \
    --wandb-entity my-org \
    --wandb-api-key "$WANDB_API_KEY" \
    --data-path ./exports/data \
    --files-path ./exports/files

  # ZenML
  uv run neptune-exporter load \
    --loader zenml \
    --data-path ./exports/data \
    --files-path ./exports/files
  ```

  > [!NOTE]
  > MLflow and W&B only accept integers. If your Neptune steps contain decimals, use the `--step-multiplier` option to convert the step values to integers. Pick a single multiplier (e.g. `1000` for millisteps) and use it consistently for all loads so that every series stays aligned.
  > Default is `1` (no scaling).

  > [!NOTE]
  > For ZenML, ensure you are logged into a ZenML server via `zenml login` before running the load command. The ZenML loader does not use `--step-multiplier` since it aggregates time-series into summary statistics rather than logging individual points. To store Neptune files in your ZenML artifact store (e.g., S3, GCS, Azure), configure a cloud [artifact store](https://docs.zenml.io/stacks/stack-components/artifact-stores) in your active stack using `zenml stack set <stack-name>` before running the load.

## Data layout on disk

- Parquet path:
  - Default: `./exports/data`
  - One directory per project, sanitized for filesystem safety (digest suffix added) but the parquet columns keep the real `project_id`/`run_id`.
  - Each run is split into `run_id_part_<n>.parquet` (Snappy-compressed). Parts roll over around 50 MB compressed.
- Files path
  - Default: `./exports/files`
  - Mirrors the sanitized project directory. File artifacts and file series are saved relative to that root.
  - Kept separate from the parquet path so you can place potentially large artifacts on different storage.

### Parquet schema

All records use `src/neptune_exporter/model.py::SCHEMA`:

| Column | Type | Description |
| --- | --- | --- |
| `project_id` | `string` | Neptune project path, in the form `workspace-name/project-name`. |
| `run_id` | `string` | Neptune run identifier.<ul><li>Neptune `3.x`: User-chosen or auto-generated, stored in `sys/custom_run_id`.</li><li>Neptune `2.x`: Auto-generated `sys/id`, e.g. `SAN-1`.</li></ul>  |
| `attribute_path` | `string` | Full attribute path. For example, `metrics/accuracy`, `metrics/loss`, `files/dataset_desc.json` |
| `attribute_type` | `string` | One of: `float`, `int`, `string`, `bool`, `datetime`, `string_set`, `float_series`, `string_series`, `histogram_series`, `file`, `file_series` |
| `step` | `decimal(18,6)` | Decimal step value, per series/metric/file series |
| `timestamp` | `timestamp(ms, UTC)` | Populated for time-based records (metrics/series/file series).  Null for parameters and files. |
| `int_value` / `float_value` / `string_value` / `bool_value` / `datetime_value` / `string_set_value` | typed columns | Payload depending on `attribute_type` |
| `file_value` | `struct{path}` | Relative path to downloaded file payload |
| `histogram_value` | `struct{type,edges,values}` | Histogram payload |

## Export flow

- Runs are listed per project and streamed in batches. Already-exported runs (those with `part_0.parquet`) are skipped so reruns are resumable.

> [!WARNING]
> Use this with care: if a run was exported and later received new data in Neptune, that new data will not be picked up unless you re-export to a fresh location.

- Data is written per run into parquet parts (~50 MB compressed per part), keeping memory usage low.
- Artifacts and file series are downloaded alongside parquet under `--files-path/<sanitized_project_id>/...`.

> [!NOTE]
> A run is considered complete once `part_0.parquet` exists. If you need a clean re-export, use a fresh `--data-path`.

## Loading flow

- Data is streamed run-by-run from parquet, using the same `--step-multiplier` to turn decimal steps into integers. Keep the multiplier consistent across loads when your Neptune steps are floats.
- **MLflow loader**:
  - Experiments are named `<project_id>/<experiment_name>`, prefixed with `--name-prefix` if provided.
  - Attribute paths are sanitized to MLflow’s key rules (alphanumeric + `_-. /`, max 250 chars).
  - Metrics/series use the integer step. Files are uploaded as artifacts from `--files-path`.
  - MLflow saves parentship/fork relationships as tags (no native forks).
- **W&B loader**:
  - Requires `--wandb-entity`. Project names derive from `project_id`, plus optional `--name-prefix`, sanitized.
  - String series become W&B Tables, histograms use `wandb.Histogram`, files/file series become artifacts. Forked runs from Neptune `3.x` are handled best-effort (W&B has limited preview support).
- **ZenML loader**:
  - Requires `zenml login` to a ZenML server. Uses ZenML's Model Control Plane to store experiment data.
  - Neptune projects become ZenML Models named `neptune-export-<project-slug>` (plus optional `--name-prefix`).
  - Neptune runs become ZenML Model Versions. Scalar attributes are logged as nested metadata for dashboard card organization.
  - Float series are aggregated into summary statistics (min/max/final/count) since ZenML doesn't have native time-series visualization.
  - Files and artifacts are uploaded via `save_artifact()` and linked to Model Versions.
- If a target run with the same name already exists in the experiment or project, the loader skips uploading that run to avoid duplicates.

## Experiment/run mapping to targets

- **MLflow:**
  - Each unique `project_id` + `sys/name` pair becomes an MLflow experiment named `<project_id>/<sys/name>` (prefixed by `--name-prefix` if provided).
  - Runs are created inside that experiment using Neptune `run_id` (or `custom_run_id` when present) as the run name. Fork relationships are ignored by MLflow.
- **W&B:**
  - Neptune `project_id` maps to the W&B project name (sanitized, plus optional `--name-prefix`).
  - `sys/name` becomes the W&B group, so all runs with the same `sys/name` land in the same group.
  - Runs are created with their Neptune `run_id` (or `custom_run_id`) as the run name. Forks from Neptune `3.x` are mapped best-effort via `fork_from`; behavior depends on W&B's fork support.
- **ZenML:**
  - Neptune `project_id` maps to a ZenML Model named `neptune-export-<org>-<project>` (plus optional `--name-prefix`).
  - Neptune's `sys/name` (experiment name) is stored as metadata and tags rather than a separate entity, since ZenML's Model Control Plane doesn't have a direct experiment concept.
  - Neptune runs become ZenML Model Versions, named after the Neptune `run_id` (or `custom_run_id`). Fork relationships are stored as metadata but not modeled natively.

## Attribute/type mapping (detailed)

The following table shows how Neptune attribute types are mapped to each target platform:

| Neptune Type | MLflow | W&B | ZenML |
|--------------|--------|-----|-------|
| `float` | Param (stringified) | Config (native float) | Metadata (native float) |
| `int` | Param (stringified) | Config (native int) | Metadata (native int) |
| `string` | Param (stringified) | Config (native string) | Metadata (native string) |
| `bool` | Param (stringified) | Config (native bool) | Metadata (native bool) |
| `datetime` | Param (stringified) | Config (ISO string) | Metadata (ISO string) |
| `string_set` | Param (comma-joined) | Config (list) | Metadata (list) |
| `float_series` | Metrics (per-point, integer step) | Metrics (per-point, integer step) | Metadata summary (min/max/final/count) |
| `string_series` | Text artifact (`series.txt`) | Text artifact (`series.txt`) | Not uploaded (metadata only) |
| `histogram_series` | JSON artifact (`histograms.json`) | `wandb.Histogram` | Not uploaded (metadata only) |
| `file` | Artifact (`log_artifact`) | Artifact (`wandb.Artifact`) | Artifact (`save_artifact`, linked to Model Version) |
| `file_series` | Artifact with step in path | Artifact with step in name | Artifact (`save_artifact`, linked to Model Version) |
| `file_set` | Artifact (directory) | Artifact (directory) | Artifact (`save_artifact`, linked to Model Version) |

**Key differences:**

- **Step handling**: MLflow and W&B require integer steps; use `--step-multiplier` to convert decimal Neptune steps. ZenML does not use steps since it aggregates series to summary statistics.
- **Attribute name sanitization**:
  - MLflow: Alphanumeric + `_-. /`, max 250 chars.
  - W&B: Must match `^[_a-zA-Z][_a-zA-Z0-9]*$`; invalid chars become `_`.
  - ZenML: Alphanumeric + `_-. /`, max 250 chars; paths are split into nested metadata for dashboard cards.
- **Series data**: MLflow and W&B log each data point individually. ZenML computes min/max/final/count summaries since the Model Control Plane doesn't have native time-series charts.

For details on Neptune attribute types, see the [documentation](https://docs.neptune.ai/attribute_types).

## Summary command

The `uv run neptune-exporter summary` command reads parquet files and prints counts of projects and runs, attribute type breakdowns, and basic step stats to help you verify the export before loading.

---

&nbsp;

> _To learn more about the Neptune acquisition and shutdown, see the [transition hub](https://docs.neptune.ai/transition_hub)._

## License

Apache 2.0. See `LICENSE.txt`.
