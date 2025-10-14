import json

from loguru import logger

from application.dataset.io import load_model_data, split_data
from application.preprocessing import build_preprocessor
from model import train_and_compare, tune_model


def autotune_pipeline(
    model_names: list[str],
    data_path: str,
    n_trials: int = 3,
    cv_folds: int = 3,
    scoring: str = "neg_mean_squared_error",
    best_model_registry_name: str = "ubereats-price-predictor",
) -> dict:
    """
    Run the end-to-end pipeline for the selected models.
    Returns the comparison results dict (and logs info).
    """
    logger.info(f"Loading data from {data_path}")
    X_train, X_test, y_train, y_test = split_data(load_model_data(data_path))

    logger.info("Building preprocessor")
    preprocessor = build_preprocessor()

    tuned_models = {}
    for model_name in model_names:
        if model_name == "lr":
            logger.info(f"Skipping tuning for {model_name}, using default params.")
            tuned_models[model_name] = {}  # or your default params if any
            continue
        logger.info(f"Starting tuning for {model_name}")
        best_params, best_metric = tune_model(
            X_train,
            y_train,
            X_test,
            y_test,
            preprocessor,
            model_name=model_name,
            n_trials=n_trials,
            cv_folds=cv_folds,
            scoring_criterion=scoring,
        )
        logger.info(f"Tuning done for {model_name}: Best Params: {best_params}, Best Metric: {best_metric}")
        tuned_models[model_name] = best_params

    logger.info("Running model comparison")
    results, best_model_name, model_uri = train_and_compare(
        tuned_models,
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor,
        scoring_criterion=scoring,
        best_model_registry_name=best_model_registry_name,
    )

    logger.info(f"Best model: {best_model_name}, URI: {model_uri}")
    logger.info(f"Comparison results:\n{json.dumps(results, indent=2)}")
    return {
        "results": results,
        "best_model_name": best_model_name,
        "model_uri": model_uri,
    }
