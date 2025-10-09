from __future__ import annotations

import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from loguru import logger
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from core.settings import settings


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    n_trials: int = 3,
    scoring_criterion: str = "neg_mean_squared_error",
) -> dict[str, int]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        }
        model = RandomForestRegressor(random_state=settings.SEED, **params)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        cv = KFold(n_splits=3, shuffle=True, random_state=settings.SEED)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring_criterion)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best RF params: {study.best_params}")
    return study.best_params, study.best_value  # type: ignore[return-value]


def plot_residuals(model, X_test, y_test, save_path=None):
    """
    Plots the residuals of the model predictions against the true values.

    Args:
    - model: The trained XGBoost model.
    - dvalid (xgb.DMatrix): The validation data in XGBoost DMatrix format.
    - valid_y (pd.Series): The true values for the validation set.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - None (Displays the residuals plot on a Jupyter window)
    """

    # Predict using the model
    preds = model.predict(X_test)

    # Calculate residuals
    residuals = y_test - preds

    # Set Seaborn style
    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    # Create scatter plot
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(y_test, residuals, color="blue", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-")

    # Set labels, title and other plot properties
    plt.title("Residuals vs True Values", fontsize=18)
    plt.xlabel("True Values", fontsize=16)
    plt.ylabel("Residuals", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")

    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    # Show the plot
    plt.close(fig)

    return fig


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            logger.info(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            logger.info(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    n_trials: int = 5,
    scoring_criterion: str = "neg_mean_squared_error",
    run_name: str = "hyperparameter-sweep-xgboost",
) -> tuple[dict[str, float | int], float]:
    """
    One MLflow parent run + a nested child run per Optuna trial.

    Trial logs (child run):
      • searched hyperparameters (your original space)
      • objective_value  -> positive MSE (= -mean(neg_MSE))  (single objective per trial)
      • cv_mean_neg, cv_std, rmse (sqrt(objective_value)), n_splits, trial_fit_time_sec
      • the trained pipeline (preprocessor + model) with a real model signature

    Parent logs:
      • best_* parameters
      • best_objective (minimized positive MSE)

    Returns: (best_params, best_value)
    """

    with mlflow.start_run(run_name=run_name):
        # log tags
        mlflow.set_tags(
            {
                "project": "Restaurant Menu Pricing",
                "run_type": "hpo",
                "model_family": "xgboost",
                "optimizer_engine": "optuna",
                "scoring": scoring_criterion,
                "n_trials": n_trials,
                "feature_set_version": 1,
            }
        )

        def objective(trial: optuna.Trial) -> float:
            # Your original search space (kept intentionally small)
            params: dict[str, float | int] = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=50),
                # uncomment any of these when you want to expand the sweep (kept minimal as requested)
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": settings.SEED,
            }

            # One child run per trial (official MLflow pattern). :contentReference[oaicite:2]{index=2}
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                mlflow.log_params(params)  # log hyperparameters. :contentReference[oaicite:3]{index=3}

                model = XGBRegressor(**params)
                pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

                # 3-fold CV on neg MSE (scikit returns negatives to keep "maximize" convention). :contentReference[oaicite:4]{index=4}
                cv = KFold(n_splits=3, shuffle=True, random_state=settings.SEED)
                t0 = time.perf_counter()
                scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring_criterion)
                fit_time = time.perf_counter() - t0

                # scores are NEGATIVE MSE -> objective is POSITIVE MSE
                cv_mean_neg = float(np.mean(scores))  # e.g., -123.4
                cv_std = float(np.std(scores))
                objective_value = -cv_mean_neg  # +123.4 (the MSE you minimize)
                rmse = float(np.sqrt(objective_value))  # convenience for eyeballing

                # Log a concise set of trial metrics (single objective + a few helpers)
                # mlflow.log_metric("objective_value", objective_value)   # main trial metric. :contentReference[oaicite:5]{index=5}
                mlflow.log_metric("cv_mse_mean", -cv_mean_neg)
                mlflow.log_metric("cv_mse_std", cv_std)
                mlflow.log_metric("cv_rmse", rmse)
                mlflow.log_metric("n_splits", cv.get_n_splits(X_train, y_train))
                mlflow.log_metric("trial_fit_time_sec", fit_time)

                # Log the trained pipeline for this trial with a proper signature. :contentReference[oaicite:6]{index=6}
                pipe.fit(X_train, y_train)

                signature_example = X_train.iloc[:10]
                signature_out = pipe.predict(signature_example)

                signature = infer_signature(signature_example, signature_out)

                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    name=f"model_pipeline_xgb_{trial.number}",  # fixed name for all trials
                    signature=signature,
                    input_example=signature_example,  # also displayed in UI
                )

                # Optional breadcrumb: link back the run id to the Optuna trial
                trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)

                return objective_value  # Optuna will minimize this

        # Initialize the Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, callbacks=[champion_callback])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mse", study.best_value)
        mlflow.log_metric("best_rmse", np.sqrt(study.best_value))

        # Optimization History Plot
        fig_history = optuna.visualization.plot_optimization_history(study)
        mlflow.log_figure(fig_history, "optimization_history.html")

        # Parameter Importance Plot
        fig_importance = optuna.visualization.plot_param_importances(study)
        mlflow.log_figure(fig_importance, "param_importances.html")

        # Parent-level: record the best
        best_params: dict[str, float | int] = study.best_params
        best_value: float = float(study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_objective", best_value)

        # Log a fit model instance
        pipe = Pipeline([("preprocessor", preprocessor), ("model", XGBRegressor(**best_params))])
        pipe.fit(X_train, y_train)

        # Log the residuals plot
        residuals = plot_residuals(pipe, X_test, y_test)
        mlflow.log_figure(figure=residuals, artifact_file="residuals.png")

        artifact_path = "best_model"

        signature_example = X_train.iloc[:10]
        signature_out = pipe.predict(signature_example)
        signature = infer_signature(signature_example, signature_out)

        mlflow.sklearn.log_model(
            sk_model=pipe,
            name=artifact_path,  # NOTE: use artifact_path (not name)
            signature=signature,
            input_example=signature_example,  # also displayed in UI
        )

        # Get the logged model uri so that we can load it from the artifact store
        model_uri = mlflow.get_artifact_uri(artifact_path)
        logger.info(f"Model Uri: {model_uri}")

    return best_params, best_value
