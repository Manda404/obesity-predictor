"""
mlflow_utils.py
=========================
Utility helpers around MLflow for experiment tracking and model lifecycle.

Features
--------
- Centralized setup using project settings (.env)
- Convenience wrappers to log params/metrics/artifacts
- Helpers to fetch best runs and experiment info
- Safe, explicit API (typed, documented) for reuse in pipelines

Author: Rostand Surel
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

from obesity_predictor.config.logger_config import logger
from obesity_predictor.config.settings import settings


def setup_mlflow() -> MlflowClient:
    """
    Initialize MLflow tracking URI and experiment from global settings.

    Returns
    -------
    MlflowClient
        A connected MLflow client ready for operations.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    logger.info(
        f"[MLflow] Tracking URI={settings.mlflow_tracking_uri} | "
        f"Experiment={settings.experiment_name}"
    )
    return MlflowClient()


@contextmanager
def mlflow_run(run_name: str):
    """
    Context manager to start/stop an MLflow run safely.

    Parameters
    ----------
    run_name : str
        The display name of the run.

    Yields
    ------
    mlflow.ActiveRun
        The active MLflow run object.
    """
    setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"[MLflow] Run started: name='{run_name}', id={run.info.run_id}")
        try:
            yield run
        finally:
            logger.info(f"[MLflow] Run ended: id={run.info.run_id}")


def log_params(params: Dict) -> None:
    """
    Log a dictionary of parameters into the current MLflow run.

    Parameters
    ----------
    params : dict
        Hyperparameters or configuration values.
    """
    if not params:
        return
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log a dictionary of metrics into the current MLflow run.

    Parameters
    ----------
    metrics : dict[str, float]
        Metric name -> value
    step : int | None
        Optional global step.
    """
    if not metrics:
        return
    mlflow.log_metrics(metrics, step=step)


def log_artifacts(artifacts: Dict[str, str]) -> None:
    """
    Log multiple artifacts with custom artifact subpaths.

    Parameters
    ----------
    artifacts : dict[str, str]
        Mapping artifact_subdir -> local_path_to_file_or_dir
    """
    for subdir, local_path in artifacts.items():
        mlflow.log_artifact(local_path, artifact_path=subdir)


def log_model_sklearn(model, artifact_path: str = "model") -> None:
    """
    Log a scikit-learn compatible model into MLflow.

    Parameters
    ----------
    model : Any
        Fitted estimator exposing predict()/fit() API.
    artifact_path : str
        Subfolder where the model will be stored.
    """
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)
    logger.success(f"[MLflow] Sklearn model logged at '{artifact_path}'")


def register_model(model_uri: str, registered_model_name: str) -> str:
    """
    Register a logged model into MLflow Model Registry.

    Parameters
    ----------
    model_uri : str
        MLflow model URI (e.g., 'runs:/<run_id>/model').
    registered_model_name : str
        Registry name for the model.

    Returns
    -------
    str
        The new registered model version string.
    """
    client = setup_mlflow()
    mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    logger.success(
        f"[MLflow] Registered model='{mv.name}' version={mv.version} from uri={model_uri}"
    )
    return mv.version


def get_experiment_id(experiment_name: str) -> Optional[str]:
    """
    Resolve an experiment name to its experiment id.

    Parameters
    ----------
    experiment_name : str

    Returns
    -------
    str | None
    """
    client = setup_mlflow()
    exp = client.get_experiment_by_name(experiment_name)
    return exp.experiment_id if exp else None


def get_best_run(
    experiment_name: str,
    metric: str = "f1_score",
    higher_is_better: bool = True,
    filter_string: str = "",
) -> Optional[Tuple[str, float]]:
    """
    Find the best run id in an experiment according to a metric.

    Parameters
    ----------
    experiment_name : str
        Experiment to search in.
    metric : str
        Metric key (as logged in MLflow).
    higher_is_better : bool
        Whether to maximize or minimize the metric.
    filter_string : str
        Optional MLflow filter (e.g., "params.model = 'XGBoost'").

    Returns
    -------
    (run_id, metric_value) | None
    """
    client = setup_mlflow()
    exp_id = get_experiment_id(experiment_name)
    if exp_id is None:
        logger.warning(f"[MLflow] Experiment '{experiment_name}' not found.")
        return None

    order = f"metrics.{metric} {'DESC' if higher_is_better else 'ASC'}"
    runs = client.search_runs(
        [exp_id],
        filter_string=filter_string,
        order_by=[order],
        max_results=1,
    )
    if not runs:
        logger.warning("[MLflow] No runs returned for best-run search.")
        return None

    run = runs[0]
    value = run.data.metrics.get(metric)
    logger.info(f"[MLflow] Best run: id={run.info.run_id}, {metric}={value}")
    return run.info.run_id, value if value is not None else float("nan")


def list_runs(
    experiment_name: str,
    order_by: Optional[Iterable[str]] = None,
    filter_string: str = "",
    max_results: int = 100,
):
    """
    List runs for a given experiment.

    Parameters
    ----------
    experiment_name : str
    order_by : Iterable[str] | None
        MLflow ordering expressions.
    filter_string : str
        MLflow filter string.
    max_results : int
        Max runs to return.

    Returns
    -------
    list[mlflow.entities.Run]
    """
    client = setup_mlflow()
    exp_id = get_experiment_id(experiment_name)
    if exp_id is None:
        return []
    return client.search_runs(
        [exp_id],
        filter_string=filter_string,
        order_by=list(order_by) if order_by else None,
        max_results=max_results,
    )
    # logger.info(f"[MLflow] Retrieved {len(runs)} runs from experiment '{experiment_name}'")