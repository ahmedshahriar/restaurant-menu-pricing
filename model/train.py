import time
from collections.abc import Mapping
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from yellowbrick.regressor import ResidualsPlot

# --- App bootstrap & settings ---
from core.settings import settings

from . import evaluate_model, get_model_spec


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
    models_with_params: Mapping[str, dict[str, int | float | str]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    cv_folds: int = 5,
    scoring_criterion: str = "neg_mean_squared_error",
    parent_run_name: str = "model-comparison",
    best_model_registry_name: str | None = None,
) -> tuple[dict[str, dict[str, float]], str, str]:
    """
    Train & compare models using provided estimators+params (fixed or tuned).
    - Child runs: params, metrics, model, residuals plot per model.
    - Parent run: comparison artifacts (boxplot, leaderboard), context, and best model tag.
    """
    results: dict[str, dict[str, float]] = {}  # model name -> metrics
    cv_rmse_all: list[np.ndarray] = []
    labels: list[str] = []
    model_uri, best_model_name = None, None

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
            {k: v for k, v in models_with_params.items()},
            artifact_file="models_params.json",
        )

        # CV strategy snapshot
        mlflow.log_dict({"n_splits": cv_folds, "shuffle": True, "random_state": settings.SEED}, "context/cv.json")

        for model_name, params in tqdm(models_with_params.items()):
            with mlflow.start_run(run_name=model_name, nested=True):
                # clone to ensure a fresh estimator for each run
                spec = get_model_spec(model_name)
                final_model = spec.build(params)

                pipe = Pipeline([("preprocessor", preprocessor), ("model", final_model)])

                logger.info(f"Fitting {model_name} model with params: {params}")

                # --- CV RMSE on the whole pipeline ---
                cv_scores = evaluate_model(
                    n_folds=cv_folds, model=pipe, X=X_train, y=y_train, scoring_criterion=scoring_criterion
                )
                cv_rmse = np.sqrt(-np.array(cv_scores))  # convert to RMSE per fold
                cv_rmse_all.append(cv_rmse)
                labels.append(model_name)

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
                batch_n = min(settings.BATCH_SIZE_INFER_TEST, len(X_test))
                t0 = time.time()
                _ = pipe.predict(X_test.iloc[:batch_n])
                infer_ms = (time.time() - t0) / batch_n * 1000
                metrics["infer_time_ms_per_row"] = float(infer_ms)

                logger.info(f"Metrics for {model_name}: {metrics}")

                results[model_name] = metrics

                # --- Log child run assets ---
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                # Residuals
                resid_path = Path(settings.ARTIFACT_DIR, f"residuals_{model_name}.png")
                _log_residuals_plot(pipe, X_train, y_train, X_test, y_test, resid_path)
                mlflow.log_artifact(str(resid_path), artifact_path=f"plots/{model_name}")

                # y_true vs y_pred (quick bias check)
                plt.figure()
                plt.scatter(y_test, y_pred, s=6)
                plt.xlabel("Ground Truth (y_true)")
                plt.ylabel("Predictions (y_pred)")
                plt.tight_layout()
                yp_path = Path(settings.ARTIFACT_DIR, f"y_true_vs_pred_{model_name}.png")
                plt.savefig(yp_path)
                plt.close()
                mlflow.log_artifact(str(yp_path), artifact_path=f"plots/{model_name}")

                signature_example = X_train.iloc[:10]
                signature_out = pipe.predict(signature_example)

                signature = infer_signature(signature_example, signature_out)
                # log model
                mlflow.sklearn.log_model(
                    pipe,
                    # name=f"model_{model_name}",
                    artifact_path=f"model_{model_name}",
                    signature=signature,
                    input_example=signature_example,
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

        # CV RMSE mean ± std bar plot
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
        if best_model_registry_name:
            # Register the child-run artifact "model_<name>" from the best child run.
            # Since we’re in the parent run, we reconstruct the path as a run-relative URI.
            best_model_name = str(best_row["model"])
            logger.info(f"Best model: {best_model_name}")
            # Find the child run with that name among the active run’s children
            # If you already know the run_id, you can pass it directly.
            try:
                client = mlflow.tracking.MlflowClient()
                children = client.search_runs(
                    experiment_ids=[parent_run.info.experiment_id],
                    filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}' and attributes.run_name = '{best_model_name}'",
                    max_results=1,
                )
                logger.info(f"Found {len(children)} run(s) for {best_model_name}")
                if not children:
                    logger.warning("No matching child run found. Nothing to register.")
                else:
                    child_run_id = children[0].info.run_id
                    model_uri = f"runs:/{child_run_id}/model_{best_model_name}"
                    # 1) Register the model (blocking wait happens inside MLflow)
                    model_version = mlflow.register_model(model_uri, name=best_model_registry_name)
                    logger.info(f"Registered '{best_model_registry_name}' v{model_version.version} from {model_uri}")

                    # 2) Best-effort: set **version** tags (safer across backends)
                    try:
                        client.set_model_version_tag(
                            name=best_model_registry_name,
                            version=model_version.version,
                            key="problem_type",
                            value="regression",
                        )
                        client.set_model_version_tag(
                            name=best_model_registry_name,
                            version=model_version.version,
                            key="model_type",
                            value=best_model_name,
                        )
                    except Exception:
                        logger.exception("Model registered, but failed to set version-level tags")

                    # 3) Best-effort: set alias if supported
                    try:
                        client.set_registered_model_alias(
                            name=best_model_registry_name, alias="champion", version=model_version.version
                        )
                    except Exception:
                        logger.warning(
                            "Model registered, but alias 'champion' could not be set (backend/version may not support aliases)"
                        )

                    logger.info(
                        f"Registered the best model {best_model_name} as '{best_model_registry_name}' (v{model_version.version})"
                    )

            except Exception:
                logger.exception("Failed to register the best model")
                # Non-fatal: leave registration best-effort
                pass

    return results, best_model_name, model_uri
