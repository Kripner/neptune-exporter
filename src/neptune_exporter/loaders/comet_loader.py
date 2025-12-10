#
# Copyright (c) 2025, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import logging
import tempfile
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Generator, Optional, Any
import pandas as pd
import pyarrow as pa

import comet_ml
from comet_ml.messages import (
    MetricMessage,
)

from neptune_exporter.types import ProjectId, TargetRunId, TargetExperimentId
from neptune_exporter.loaders.loader import DataLoader

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def is_image(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS


class CometLoader(DataLoader):
    """Loads Neptune data from parquet files into a Comet installation."""

    def __init__(
        self,
        workspace: str,
        api_key: Optional[str] = None,
        name_prefix: Optional[str] = None,
        show_client_logs: bool = False,
    ):
        """
        Initialize Comet loader.

        Args:
            workspace: Comet workspace
            api_key: Optional Comet API key for authentication
            name_prefix: Optional prefix for project and run names
            verbose: Enable verbose logging
        """
        self.workspace = workspace
        self.name_prefix = name_prefix
        self._logger = logging.getLogger(__name__)
        self._comet_experiment: Optional[comet_ml.Experiment] = None

        # Configure Comet logging - suppress INFO and WARNING messages
        # The logger name is "comet_ml" (not "COMET" - that's just the formatter prefix)
        if show_client_logs:
            logging.getLogger("comet_ml").setLevel(logging.INFO)
        else:
            logging.getLogger("comet_ml").setLevel(logging.ERROR)

    def _sanitize_attribute_name(self, attribute_path: str) -> str:
        """
        Sanitize Neptune attribute path.

        Key constraints:
        - Must start with a letter or underscore
        - Can only contain letters, numbers, and underscores
        - Pattern: /^[_a-zA-Z][_a-zA-Z0-9]*$/
        """
        # Replace invalid characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", attribute_path)

        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "_" + sanitized

        # Handle empty result
        if not sanitized:
            sanitized = "_attribute"

        return sanitized

    def _get_project_name(self, project_id: str) -> str:
        """Get Comet project name from Neptune project ID."""
        name = project_id

        if self.name_prefix:
            name = f"{self.name_prefix}_{name}"

        # Sanitize project name (alphanumeric, hyphens, underscores)
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        return name

    def _convert_step_to_int(self, step: Decimal, step_multiplier: int) -> int:
        """Convert Neptune decimal step to integer step."""
        if step is None:
            return 0
        return int(float(step) * step_multiplier)

    # Overloaded methods:

    def create_experiment(
        self, project_id: str, experiment_name: str
    ) -> TargetExperimentId:
        """
        Neptune experiment_name maps to Comet experiment.
        We return the experiment name to use.
        """
        return TargetExperimentId(experiment_name)

    def find_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        """
        Find an experiment by name in a Comet project.

        Args:
            run_name: Name of the run to find
            experiment_id: comet experiment.id
            project_id: Comet project_name

        Returns:
            None, as Comet doesn't support resuming or forking runs
        """
        return None

    def create_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId] = None,
        parent_run_id: Optional[TargetRunId] = None,
        fork_step: Optional[float] = None,
        step_multiplier: Optional[int] = None,
    ) -> TargetRunId:
        """
        Create a Comet experiment.

        Args:
            fork_step: Fork step as float (decimal). Will be converted to int using step_multiplier.
            step_multiplier: Step multiplier for converting decimal steps to integers.
                If provided, will be used for fork_step conversion. If not provided,
                will calculate from fork_step alone as fallback.
        """
        project_name = self._get_project_name(project_id)

        try:
            self._comet_experiment = comet_ml.Experiment(
                workspace=self.workspace,
                project_name=project_name,
                experiment_name=run_name,
                log_code=False,
                log_graph=False,
                auto_param_logging=False,
                parse_args=False,
                auto_output_logging=None,
                log_env_details=False,
                log_git_metadata=False,
                log_git_patch=False,
                log_env_gpu=False,
                log_env_host=False,
                log_env_cpu=False,
                log_env_network=False,
                log_env_disk=False,
                display_summary_level=0,
            )
            self._comet_experiment.set_name(run_name)

            self._logger.info(
                f"Created Comet experiment '{run_name}' with ID {self._comet_experiment.id}"
            )
            return TargetRunId(self._comet_experiment.id)

        except Exception:
            self._logger.error(
                f"Error creating project {project_id}, run '{run_name}'",
                exc_info=True,
            )
            raise

    def upload_run_data(
        self,
        run_data: Generator[pa.Table, None, None],
        run_id: TargetRunId,
        files_directory: Path,
        step_multiplier: int,
    ) -> None:
        """Upload all data for a single experiment to Comet.

        Args:
            step_multiplier: Step multiplier for converting decimal steps to integers
        """
        try:
            for run_data_part in run_data:
                run_df = run_data_part.to_pandas()

                self.upload_parameters(run_df, run_id)
                self.upload_metrics(run_df, run_id, step_multiplier)
                self.upload_artifacts(run_df, run_id, files_directory, step_multiplier)
                self._comet_experiment.end()

            self._logger.info(f"Successfully uploaded run {run_id} to Comet")

        except Exception:
            self._logger.error(f"Error uploading data for run {run_id}", exc_info=True)
            raise

    def upload_parameters(self, run_data: pd.DataFrame, run_id: TargetRunId) -> None:
        """Upload parameters to Comet experiment."""
        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")

        param_types = {"float", "int", "string", "bool", "datetime", "string_set"}
        param_data = run_data[run_data["attribute_type"].isin(param_types)]

        if param_data.empty:
            return

        parameters = {}
        for _, row in param_data.iterrows():
            attr_name = self._sanitize_attribute_name(row["attribute_path"])

            # Get the appropriate value based on attribute type
            if row["attribute_type"] == "float" and pd.notna(row["float_value"]):
                parameters[attr_name] = row["float_value"]
            elif row["attribute_type"] == "int" and pd.notna(row["int_value"]):
                parameters[attr_name] = int(row["int_value"])
            elif row["attribute_type"] == "string" and pd.notna(row["string_value"]):
                if "model_summary" in attr_name:
                    self._comet_experiment.set_model_graph(row["string_value"])
                else:
                    parameters[attr_name] = row["string_value"]
            elif row["attribute_type"] == "bool" and pd.notna(row["bool_value"]):
                parameters[attr_name] = bool(row["bool_value"])
            elif row["attribute_type"] == "datetime" and pd.notna(
                row["datetime_value"]
            ):
                parameters[attr_name] = str(row["datetime_value"])
            elif (
                row["attribute_type"] == "string_set"
                and row["string_set_value"] is not None
            ):
                parameters[attr_name] = list(row["string_set_value"])

        if parameters:
            self._comet_experiment.log_parameters(parameters)
            self._logger.info(f"Uploaded {len(parameters)} parameters for run {run_id}")

    def upload_metrics(
        self, run_data: pd.DataFrame, run_id: TargetRunId, step_multiplier: int
    ) -> None:
        """Upload metrics (float series) to Comet experiment.

        Args:
            step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active experiment")

        metrics_data = run_data[run_data["attribute_type"] == "float_series"]

        if metrics_data.empty:
            return

        # Use global step multiplier (calculated from all series + fork_step)
        # Group by step to log all metrics at each step together
        for row in metrics_data.itertuples():
            step_value = row.step
            timestamp = row.timestamp.timestamp()
            if pd.notna(step_value):
                step = self._convert_step_to_int(step_value, step_multiplier)
                if pd.notna(row.float_value):
                    metric_name = self._sanitize_attribute_name(row.attribute_path)
                    value = row.float_value
                    self._log_metric(metric_name, value, step=step, timestamp=timestamp)

        self._logger.info(f"Uploaded metrics for run {run_id}")

    def _log_metric(self, name, value, step, timestamp):
        """
        Add the metric with timestamp to an Comet Experiment
        """
        message = MetricMessage(
            context=None,
            timestamp=timestamp,
        )
        message.set_metric(name, value, step=step)
        self._comet_experiment._enqueue_message(message)

    def upload_artifacts(
        self,
        run_data: pd.DataFrame,
        run_id: TargetRunId,
        files_base_path: Path,
        step_multiplier: int,
    ) -> None:
        """Upload files and series as assets to Comet experiment.

        Args:
            step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)
        """
        if self._comet_experiment is None:
            raise RuntimeError("No active run")

        # Handle regular files
        file_data = run_data[
            run_data["attribute_type"].isin(["file", "file_set", "artifact"])
        ]
        for _, row in file_data.iterrows():
            if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                file_path = files_base_path / row["file_value"]["path"]
                if file_path.exists():
                    attr_name = self._sanitize_attribute_name(row["attribute_path"])
                    if file_path.is_file():
                        self._comet_experiment.log_asset(
                            file_data=file_path,
                            file_name=attr_name,
                        )
                    else:
                        if "source_code" in str(file_path):
                            # Go through the folder
                            for filename in file_path.iterdir():
                                if filename.is_file():
                                    if filename.suffix == ".zip":
                                        # Unzip the file and log each file with code_name from zipfile
                                        with tempfile.TemporaryDirectory() as temp_dir:
                                            with zipfile.ZipFile(
                                                filename, "r"
                                            ) as zip_ref:
                                                # Get all file paths in the zip
                                                zip_file_paths = zip_ref.namelist()
                                                zip_ref.extractall(temp_dir)
                                                # Log each file with its path from the zip as code_name
                                                for zip_file_path in zip_file_paths:
                                                    # Skip directories
                                                    if not zip_file_path.endswith("/"):
                                                        extracted_file = (
                                                            Path(temp_dir)
                                                            / zip_file_path
                                                        )
                                                        if extracted_file.is_file():
                                                            self._comet_experiment.log_code(
                                                                file_name=str(
                                                                    extracted_file
                                                                ),
                                                                code_name=zip_file_path,
                                                            )
                                    else:
                                        # Log the file with code_name from attr_name
                                        self._comet_experiment.log_code(
                                            file_name=str(filename), code_name=attr_name
                                        )
                        else:
                            self._comet_experiment.log_asset_folder(
                                folder=file_path,
                            )
                else:
                    self._logger.warning(f"File not found: {file_path}")

        # Handle file series
        file_series_data = run_data[run_data["attribute_type"] == "file_series"]
        for attr_path, group in file_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            for _, row in group.iterrows():
                if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                    file_path = files_base_path / row["file_value"]["path"]
                    if file_path.exists():
                        step = (
                            self._convert_step_to_int(row["step"], step_multiplier)
                            if pd.notna(row["step"])
                            else 0
                        )
                        if file_path.is_file():
                            if is_image(file_path):
                                self._comet_experiment.log_image(
                                    image_data=file_path,
                                    name=attr_name,
                                    step=step,
                                )
                            else:
                                self._comet_experiment.log_asset(
                                    file_data=file_path,
                                    file_name=attr_name,
                                    step=step,
                                )
                        else:
                            self._comet_experiment.log_asset_folder(
                                folder=file_path,
                                step=step,
                            )
                    else:
                        self._logger.warning(f"File not found: {file_path}")

        # Handle string series as text asset
        string_series_data = run_data[run_data["attribute_type"] == "string_series"]
        for attr_path, group in string_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            # Create temporary file with text content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", encoding="utf-8"
            ) as tmp_file:
                for _, row in group.iterrows():
                    if pd.notna(row["string_value"]):
                        series_step = (
                            self._convert_step_to_int(row["step"], step_multiplier)
                            if pd.notna(row["step"])
                            else None
                        )
                        timestamp = (
                            row["timestamp"].isoformat()
                            if pd.notna(row["timestamp"])
                            else None
                        )
                        text_line = (
                            f"{series_step}; {timestamp}; {row['string_value']}\n"
                        )
                        tmp_file.write(text_line)
                tmp_file_path = tmp_file.name
                self._comet_experiment.log_asset(
                    file_data=tmp_file_path,
                    file_name=attr_name,
                    step=series_step,
                )

        # Handle histogram series as Comet Histograms
        histogram_series_data = run_data[
            run_data["attribute_type"] == "histogram_series"
        ]
        for attr_path, group in histogram_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)
            # Use global step multiplier

            for _, row in group.iterrows():
                if pd.notna(row["histogram_value"]) and isinstance(
                    row["histogram_value"], dict
                ):
                    breakpoint()
                    step = (
                        self._convert_step_to_int(row["step"], step_multiplier)
                        if pd.notna(row["step"])
                        else 0
                    )
                    hist = row["histogram_value"]

                    # Convert Neptune histogram to Comet Histogram
                    # Neptune format: {"type": str, "edges": list, "values": list}
                    try:
                        print("TODO: log histogram")
                    except Exception:
                        self._logger.error(
                            f"Failed to log histogram for {attr_path} at step {step}",
                            exc_info=True,
                        )

        self._logger.info(f"Uploaded assets for run {run_id}")
