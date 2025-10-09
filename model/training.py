import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from yellowbrick.regressor import ResidualsPlot

# --- App bootstrap & settings ---
from application.config.bootstrap import apply_global_settings
from core.settings import settings

apply_global_settings()


def _log_residuals_plot(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_path: Path,
) -> Path:
    """Save a ResidualsPlot for the fitted pipeline."""

    # https://www.scikit-yb.org/en/latest/api/regressor/residuals.html

    # # Instantiate the linear model and visualizer
    # viz = ResidualsPlot(pipeline)  # hist=False, qqplot=True
    #
    # viz.fit(X_train, y_train)  # Fit the training data to the visualizer
    # viz.score(X_test, y_test)  # E

    # Fit/score using transformed features for the final regressor
    pre = pipeline.named_steps["preprocessor"]
    reg = pipeline.named_steps["model"]

    X_train_t = pre.fit_transform(X_train, y_train)
    X_test_t = pre.transform(X_test)

    viz = ResidualsPlot(reg)
    viz.fit(X_train_t, y_train)
    viz.score(X_test_t, y_test)
    viz.show(outpath=str(out_path), clear_figure=True)
    plt.close("all")
    return out_path


def _boxplot_cv_rmse(results: list[np.ndarray], labels: list[str], out_path: Path) -> Path:
    """Save a compact CV RMSE comparison boxplot."""
    plt.boxplot(results, labels=labels, showmeans=True)
    plt.xlabel("Models")
    plt.ylabel("CV RMSE (lower is better)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def train_and_compare(
    models_with_params: Mapping[str, tuple[BaseEstimator, dict[str, Any]]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    cv_folds: int = 5,
    parent_run_name: str = "model-comparison",
    registered_best_model: str | None = None,  # e.g., "ubereats-price-regressor"
) -> dict[str, dict[str, float]]:
    """
    Train & compare models using provided estimators+params (fixed or tuned).
    - Child runs: params, metrics, model, residuals plot per model.
    - Parent run: comparison artifacts (boxplot, leaderboard), context, and best model tag.
    """
    results: dict[str, dict[str, float]] = {}  # model name → metrics
    cv_rmse_all: list[np.ndarray] = []
    labels: list[str] = []

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        # ---- Parent context logs (simple, useful) ----
        mlflow.set_tags({"run_type": "comparison"})
        mlflow.log_params(
            {
                "n_models": len(models_with_params),
                "cv_folds": cv_folds,
                "train_rows": int(X_train.shape[0]),
                "test_rows": int(X_test.shape[0]),
            }
        )
        # Log models+params mapping for traceability
        mlflow.log_dict(
            {k: v[1] for k, v in models_with_params.items()},
            artifact_file="models_params.json",
        )

        # CV strategy snapshot
        mlflow.log_dict({"n_splits": cv_folds, "shuffle": True, "random_state": settings.SEED}, "context/cv.json")

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=settings.SEED)

        for name, (estimator, params) in models_with_params.items():
            with mlflow.start_run(run_name=name, nested=True):
                # clone to ensure a fresh estimator for each run
                est: BaseEstimator = clone(estimator)
                if params:
                    est.set_params(**params)

                pipe = Pipeline([("preprocessor", preprocessor), ("model", est)])

                logger.info(f"Fitting {name} model with params: {est.get_params()}")

                # --- CV RMSE on the whole pipeline ---
                cv_scores = cross_val_score(
                    pipe,
                    X_train,
                    y_train,
                    scoring="neg_mean_squared_error",
                    cv=kf,
                    n_jobs=-1,
                    error_score="raise",
                )
                cv_rmse = np.sqrt(-cv_scores)
                cv_rmse_all.append(cv_rmse)
                labels.append(name)

                # --- Fit / timings ---
                t0 = time.time()
                pipe.fit(X_train, y_train)
                train_seconds = time.time() - t0

                # --- metrics on test data ---
                y_pred = pipe.predict(X_test)
                metrics = {
                    "MAE": float(mean_absolute_error(y_test, y_pred)),
                    "RMSE": float(root_mean_squared_error(y_test, y_pred)),
                    "R2": float(r2_score(y_test, y_pred)),
                    "CV_RMSE_mean": float(cv_rmse.mean()),
                    "CV_RMSE_std": float(cv_rmse.std()),
                    "train_seconds": float(train_seconds),
                }

                # simple inference latency on a small batch
                batch_n = min(256, len(X_test))
                t0 = time.time()
                _ = pipe.predict(X_test.iloc[:batch_n])
                infer_ms = (time.time() - t0) / batch_n * 1000
                metrics["avg_infer_ms_per_row"] = float(infer_ms)

                logger.info(f"Metrics for {name}: {metrics}")

                results[name] = metrics

                # --- Log child run assets ---
                mlflow.log_params(est.get_params())
                mlflow.log_metrics(metrics)

                # Residuals
                resid_path = Path(settings.ARTIFACT_DIR, f"residuals_{name}.png")
                _log_residuals_plot(pipe, X_train, y_train, X_test, y_test, resid_path)
                mlflow.log_artifact(str(resid_path), artifact_path=f"plots/{name}")

                # y_true vs y_pred (quick bias check)
                plt.figure()
                plt.scatter(y_test, y_pred, s=6)
                plt.xlabel("Ground Truth (y_true)")
                plt.ylabel("Predictions (y_pred)")
                plt.tight_layout()
                yp_path = Path(settings.ARTIFACT_DIR, f"y_true_vs_pred_{name}.png")
                plt.savefig(yp_path)
                plt.close()
                mlflow.log_artifact(str(yp_path), artifact_path=f"plots/{name}")

                signature_example = X_train.iloc[:10]
                signature_out = pipe.predict(signature_example)

                signature = infer_signature(signature_example, signature_out)
                # log model
                mlflow.sklearn.log_model(
                    pipe, name=f"model_{name}", signature=signature, input_example=signature_example
                )

        # ---- Parent-level comparison artifacts ----
        # Existing CV RMSE boxplot
        cmp_path = Path(settings.ARTIFACT_DIR, "cv_rmse_comparison.png")
        _boxplot_cv_rmse(cv_rmse_all, labels, cmp_path)
        mlflow.log_artifact(str(cmp_path), artifact_path="plots/comparison")

        # Leaderboard CSV/JSON (sorted by CV_RMSE_mean)
        # generate leaderboard DataFrame
        leaderboard = (
            pd.DataFrame.from_dict(results, orient="index")
            .assign(model=lambda d: d.index)
            .sort_values("CV_RMSE_mean")
            .reset_index(drop=True)
        )

        # feed leaderboard artifacts
        lb_csv = Path(settings.ARTIFACT_DIR, "leaderboard.csv")
        lb_json = Path(settings.ARTIFACT_DIR, "leaderboard.json")

        leaderboard.to_csv(lb_csv, index=False)
        leaderboard.to_json(lb_json, orient="records", indent=2)

        mlflow.log_artifact(str(lb_csv), artifact_path="tables")
        mlflow.log_artifact(str(lb_json), artifact_path="tables")

        # Simple bar chart for CV_RMSE_mean
        plt.bar(leaderboard["model"], leaderboard["CV_RMSE_mean"])
        plt.ylabel("CV RMSE mean (lower is better)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        bar_path = Path(settings.ARTIFACT_DIR, "cv_rmse_mean_bar.png")
        plt.savefig(bar_path)
        plt.close()
        mlflow.log_artifact(str(bar_path), artifact_path="plots/comparison")

        means = leaderboard["CV_RMSE_mean"].values
        stds = leaderboard["CV_RMSE_std"].values
        models = leaderboard["model"].values
        plt.bar(models, means, yerr=stds, capsize=5)
        plt.ylabel("CV RMSE (mean ± std) — lower is better")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        bar_path = Path(settings.ARTIFACT_DIR, "cv_rmse_mean_bar.png")
        plt.savefig(bar_path)
        plt.close()
        mlflow.log_artifact(str(bar_path), artifact_path="plots/comparison")

        # Raw CV arrays (re-plot later if needed)
        cv_dump = {lbl: arr.tolist() for lbl, arr in zip(labels, cv_rmse_all, strict=False)}
        mlflow.log_dict(cv_dump, artifact_file="cv_rmse_raw.json")

        # Tag best model on the parent run
        best_row = leaderboard.iloc[0]
        mlflow.set_tags(
            {
                "best_model": str(best_row["model"]),
                "best_model_cv_rmse_mean": f"{best_row['CV_RMSE_mean']:.6f}",
                "best_model_cv_rmse_std": f"{best_row['CV_RMSE_std']:.6f}",
            }
        )

        # ---- register the best model from the comparison ----
        if registered_best_model:
            # Register the child-run artifact "model_<name>" from the best child run.
            # Since we’re in the parent run, we reconstruct the path as a run-relative URI.
            best_name = str(best_row["model"])
            logger.info(f"Best model: {best_name}")
            # Find the child run with that name among the active run’s children
            # If you already know the run_id, you can pass it directly.
            try:
                client = mlflow.tracking.MlflowClient()
                children = client.search_runs(
                    experiment_ids=[parent_run.info.experiment_id],
                    filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}' and attributes.run_name = '{best_name}'",
                    max_results=1,
                )
                logger.info(f"Found {len(children)} runs")
                if children:
                    child_run_id = children[0].info.run_id
                    model_uri = f"runs:/{child_run_id}/model_{best_name}"
                    mlflow.register_model(model_uri, name=registered_best_model)
            except Exception:
                logger.exception("Failed to register the best model")
                # Non-fatal: leave registration best-effort
                pass

    return results
