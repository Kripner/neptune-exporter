# Neptune Exporter Project - Data Model Challenges

## MLflow Model

- **organisation/project** → such division does not seem to exist
  - we can just add it to the experiment name
  - or ask user for an (optional) prefix for each project (as a cli param) and import runs as they are in the data loading step

- **experiment** → experiment
- **run** → run

- **forks** → nested runs
  - mlflow supports nesting runs under a parent. It does not seem to inherit series. The runs are nested in the UI instead.
  - The other option would be to duplicate the data to all forks and save them as independent runs, but it could generate much more data on the mlflow side.

- **attribute path** → attribute key
  - mlflow key constraints: This string may only contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/). All backend stores will support keys up to length 250, but some may support larger keys.
  - neptune path constraints: MAX_ATTRIBUTE_PATH_LENGTH = 1024
  - neptune accepts much longer paths, especially if we additionally attempt to prepend them with the project... could attempt to contract the name and hope that users rarely use paths longer than 250.

- **configs** → params
  - neptune: float, int, string, bool, datetime
  - mlflow: all values are stringified
  - Since all values are stringified, uploading our values won't be a problem. Recreating our model from MLflow would be though.

- **tags** → tags
  - mlflow has tags. Use mlflow.set_tag or set on start_run

- **metrics (float series)** → metrics
  - neptune uses steps - decimal(18, 6)
  - mlflow only accepts steps as ints
  - It would be generally possible to transform our decimals into ints with a simple *1_000_000, but if someone uses less precision they may not be satisfied with such a result and would prefer a different transformation

- **files** → artifacts
  - It seems possible to simply upload files from a given local path, not sure if there are some limits that are different, probably depends on the file storage behind both systems

- **file series** → artifacts?
  - mlflow does not allow to save a file per step. I guess we could append our step to the artifact name.

- **string series** → artifacts?
  - mlflow does not have a direct equivalent. It has log_text (Log text as an artifact), but we would have to encode our steps in the string series. Or log_table, which could retain a two-column structure.

- **histogram series** → artifacts?
  - mlflow does not have an equivalent. It's worse in the case of this type, than with other series, because it won't properly display the histograms. Again, we could save them as artifacts using log_table.

## W&B Model

- **organisation/project** → entity/project

- **experiment** → experiment
- **run** → run

- **forks** → forks
  - w&b allows creation of forks using wandb.init(fork_from=f"{id}?_step={step}")

- **attribute path** → metric/config names
  - w&b:
    - Allowed characters: Letters (A-Z, a-z), digits (0-9), and underscores (_)
    - Starting character: Names must start with a letter or underscore
    - Pattern: Metric names should match `/^[_a-zA-Z][_a-zA-Z0-9]*$/`

- **configs** → config
  - neptune: float, int, string, bool, datetime
  - w&b: can't find an exact list but supports various scalars (int, float and string), will be easy to save our other types even if indirectly
  - See https://docs.wandb.ai/guides/track/create-an-experiment/#capture-a-dictionary-of-hyperparameters

- **tags** → tags

- **metrics (float series)** → log floats
  - neptune uses steps - decimal(18, 6)
  - w&b uses steps - ints
  - It would be generally possible to transform our decimals into ints with a simple *1_000_000, but if someone uses less precision they may not be satisfied with such a result and would prefer a different transformation
  - See https://docs.wandb.ai/ref/python/experiments/run/#method-runlog

  - w&b run.log supports scalars (int, float, string), images, video, audio, histograms, plots, html, and tables - meaning our float/string/histogram series should be covered

- **string series** → log strings
- **histogram series** → log histrograms

- **files** → artifacts (type=unspecified)
  - See https://docs.wandb.ai/guides/artifacts/

- **file series** → artifacts?
  - artifacts do not seem to be associated with a step. Specific files like images, videos, etc. can be, but it'd be risky to map our files to either metrics or artifacts depending on their mimetype. Probably should just include the step in the artifact name.

- **source code** → code
  - neptune optionally logs some source code (under path `source_code/`). It would be a UI improvement to log it as code using `log_code`
