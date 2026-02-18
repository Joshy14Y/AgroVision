from __future__ import annotations
import os
from typing import Optional, Literal, Any, Dict, Union

from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow(
    *,
    tracking_env_var: str = "MLFLOW_SERVER_URI",
    experiment_env_var: str = "MLFLOW_EXPERIMENT_NAME",
    default_experiment: str = "default",
    default_tags: Optional[Dict[str, Any]] = None,
) -> mlflow:

    load_dotenv()

    tracking_uri = os.getenv(tracking_env_var)
    if not tracking_uri:
        raise RuntimeError(
            f"{tracking_env_var} not set. Add it to your .env (e.g., http://<ip>:8050)."
        )

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv(experiment_env_var, default_experiment)
    mlflow.set_experiment(experiment_name)

    if default_tags:
        mlflow.set_tags(default_tags)

    return mlflow


def load_model(
    model_ref: str,
    *,
    flavor: Optional[
        Literal[
            "pyfunc",
            "sklearn",
            "keras",
            "pytorch",
            "xgboost",
            "lightgbm",
            "catboost",
            "statsmodels",
            "tensorflow"
        ]
    ] = None,
    device: Union[Literal["auto","cpu","cuda","mps"], str] = "auto"
):
    """
    Load a model from MLflow given a model reference.

    Accepts any of:
      - runs:/<RUN_ID>/model
      - models:/<MODEL_NAME>/<STAGE>         (Production, Staging)
      - models:/<MODEL_NAME>/<VERSION>       (1, 2, ...)
      - file:///... or s3://...              (advanced/direct)

    If 'flavor' is None, uses generic pyfunc (works for any logged flavor with a python
    loader). If you know the exact flavor you want, pass it for native types (e.g., sklearn).
    """

    if flavor in (None, "pyfunc"):
        return mlflow.pyfunc.load_model(model_uri=model_ref, map_location=device)

    # Native flavor loaders (return native model objects)
    if flavor == "sklearn":
        return mlflow.sklearn.load_model(model_uri=model_ref, map_location=device)
    if flavor == "keras" or flavor == "tensorflow":
        return mlflow.keras.load_model(model_uri=model_ref, map_location=device)
    if flavor == "pytorch":
        return mlflow.pytorch.load_model(model_uri=model_ref, map_location=device)
    if flavor == "xgboost":
        return mlflow.xgboost.load_model(model_uri=model_ref, map_location=device)
    if flavor == "lightgbm":
        return mlflow.lightgbm.load_model(model_uri=model_ref, map_location=device)
    if flavor == "catboost":
        return mlflow.catboost.load_model(model_uri=model_ref, map_location=device)
    if flavor == "statsmodels":
        return mlflow.statsmodels.load_model(model_uri=model_ref, map_location=device)

    raise ValueError(f"Unsupported flavor: {flavor}")