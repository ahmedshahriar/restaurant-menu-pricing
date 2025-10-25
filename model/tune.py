from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from core.settings import settings
from model.registry import get_model_spec

from . import evaluate_model


def _pred_vs_true_figure(y_true: pd.Series, y_pred: np.ndarray, title: str = "Predicted vs True"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.35)
    m = float(min(float(np.min(y_true)), float(np.min(y_pred))))
    M = float(max(float(np.max(y_true)), float(np.max(y_pred))))
    ax.plot([m, M], [m, M], linestyle="--")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_residuals(model, X_test, y_test, save_path=None):
    """
    Plots the residuals of the model predictions against the true values.

    Args:
    - model: The trained  model.
    - X_test: The feature set for the validation set.
    - y_test: The true values for the validation set.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - None (Displays the residuals plot on a Jupyter window)
    """

    # Predict using the model
    preds = model.predict(X_test)

    # Calculate residuals
    residuals = y_test - preds

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
        plt.savefig(save_path, format="png", dpi=300)

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


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    model_name: str,
    n_trials: int = 5,
    cv_folds: int = 5,
    scoring_criterion: str = "neg_mean_squared_error",
) -> tuple[dict[str, float | int], float]:
    """
    One MLflow parent run + a nested child run per Optuna trial.

    Trial logs (child run):
      • searched hyperparameters (given hyperparameter space)
      • objective_value  -> positive MSE (= -mean(neg_MSE))  (single objective per trial)
      • cv_mse_mean, cv_mse_std, rmse (sqrt(objective_value)), n_splits, trial_fit_time_sec
      • the trained pipeline (preprocessor + model) with a real model signature

    Parent logs:
      • best_* parameters
      • best_objective (minimized positive MSE)

    Returns: (best_params, best_value)
    """

    model_spec = get_model_spec(model_name)  # e.g., "rforest" or "xgboost"

    with mlflow.start_run(run_name=f"{model_name}-tuning"):
        # log tags
        mlflow.set_tags(
            {
                "project": "Restaurant Menu Pricing",
                "run_type": "hpo",
                "model_family": f"{model_name}",
                "optimizer_engine": "optuna",
                "scoring": scoring_criterion,
                "n_trials": n_trials,
                "cv_folds": cv_folds,
                "feature_set_version": 1,
            }
        )

        trial_rows: list[dict] = []

        def objective(trial: optuna.Trial) -> float:
            # One child run per trial (official mlflow pattern)
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("model_flavor", "sklearn")
                # 1) sample hyperparameters from the spec's space
                params = model_spec.param_space(trial)
                # 2) build the estimator with base kwargs merged with trial params
                model = model_spec.build(params)

                mlflow.log_params(params)

                # model = XGBRegressor(**params)
                pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

                t0 = time.perf_counter()
                scores = evaluate_model(
                    n_folds=cv_folds, model=pipe, X=X_train, y=y_train, scoring_criterion=scoring_criterion
                )
                fit_time = time.perf_counter() - t0

                # scores are NEGATIVE MSE -> objective is POSITIVE MSE
                cv_mse_mean = float(np.mean(scores))  # e.g., -123.4
                cv_mse_std = float(np.std(scores))
                objective_value = -cv_mse_mean  # +123.4 (the MSE you minimize)
                cv_rmse_mean = float(np.mean(np.sqrt([-s for s in scores])))

                # Log a concise set of trial metrics (single objective + a few helpers)
                mlflow.log_metric("objective_value", objective_value)
                mlflow.log_metric("cv_mse_mean", -cv_mse_mean)  # positive MSE
                mlflow.log_metric("cv_mse_std", cv_mse_std)
                mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
                mlflow.log_metric("n_splits", cv_folds)
                mlflow.log_metric("trial_fit_time_sec", fit_time)

                # Log the trained pipeline for this trial with a proper signature.
                pipe.fit(X_train, y_train)

                example_in = X_train.iloc[:10]
                signature_ml = infer_signature(example_in, pipe.predict(example_in))

                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    # name=f"model_pipeline_{model_name}_{trial.number}",
                    artifact_path=f"model_pipeline_{model_name}_{trial.number}",
                    signature=signature_ml,
                    input_example=example_in,  # also displayed in UI
                )

                # Trial-level figure (Pred vs True on held-out test)
                y_pred_test = pipe.predict(X_test)
                fig = _pred_vs_true_figure(
                    y_true=y_test, y_pred=y_pred_test, title=f"{model_name}-Trial {trial.number} – Pred vs True (test)"
                )
                mlflow.log_figure(fig, f"plots/{model_name}_pred_vs_true_test.png")
                plt.close(fig)

                # keep a compact table row for parent summary CSV
                trial_rows.append(
                    {
                        "trial": trial.number,
                        **params,
                        "objective_value": objective_value,
                        "cv_rmse_mean": cv_rmse_mean,
                        "cv_mse_mean": -cv_mse_mean,  # positive MSE
                        "cv_mse_std": cv_mse_std,
                        "fit_time_sec": fit_time,
                    }
                )

                # Optional breadcrumb: link back the run id to the Optuna trial
                trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)

                return objective_value  # Optuna will minimize this

        # Initialize the Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[champion_callback])

        # ----- Parent-level logging -----
        # Best params & metrics
        best_params: dict[str, float | int] = study.best_params
        best_value: float = float(study.best_value)

        # Params: prefix to avoid collisions
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        # Metrics
        mlflow.log_metric("best_mse", best_value)
        mlflow.log_metric("best_objective_value", best_value)
        mlflow.log_metric("best_rmse", np.sqrt(best_value))

        # Useful tag
        mlflow.set_tag("best_trial", study.best_trial.number)

        # Trials summary CSV (quick compare at parent level)
        if trial_rows:
            # generate leaderboard DataFrame
            leaderboard = (
                pd.DataFrame(trial_rows).assign(model=model_name).sort_values("objective_value").reset_index(drop=True)
            )

            # feed leaderboard artifacts
            lb_csv = Path(settings.ARTIFACT_DIR, f"{model_name}_trials_summary.csv")
            lb_json = Path(settings.ARTIFACT_DIR, f"{model_name}_trials_summary.json")

            leaderboard.to_csv(lb_csv, index=False)
            leaderboard.to_json(lb_json, orient="records", indent=2)

            mlflow.log_artifact(str(lb_csv), artifact_path="tables")
            mlflow.log_artifact(str(lb_json), artifact_path="tables")

        # Optimization Plot
        # Optimization History Plot
        fig_history = optuna.visualization.plot_optimization_history(study)
        mlflow.log_figure(fig_history, "plots/optimization_history.html")

        # Optimization Hyperparameter Importance Plot
        fig_importances = optuna.visualization.plot_param_importances(study)
        mlflow.log_figure(fig_importances, "plots/param_importances.html")

        # Log a fit model instance
        spec = get_model_spec(model_name)
        final_model = spec.build(best_params)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", final_model)])
        pipe.fit(X_train, y_train)

        # Log the residuals plot
        residuals = plot_residuals(pipe, X_test, y_test)
        mlflow.log_figure(figure=residuals, artifact_file="plots/residuals.png")

        artifact_path = f"best_model_{model_name}"

        signature_example = X_train.iloc[:10]
        signature_out = pipe.predict(signature_example)
        signature = infer_signature(signature_example, signature_out)

        mlflow.sklearn.log_model(
            sk_model=pipe,
            # name=artifact_path,
            artifact_path=artifact_path,
            signature=signature,
            input_example=signature_example,
        )

        # Get the logged model uri so that we can load it from the artifact store
        model_uri = mlflow.get_artifact_uri(artifact_path)
        logger.info(f"Model Uri: {model_uri}")

    return best_params, best_value
