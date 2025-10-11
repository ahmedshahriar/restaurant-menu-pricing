import click
import mlflow
from loguru import logger

from application.config import apply_global_settings
from core import settings
from model import REGISTRY
from pipelines import autotune_pipeline


def _validate_model_names(ctx, param, value):
    """
    click callback to validate the `--models` input.
    `value` is a comma-separated string (or None).
    Returns a list of model names.
    """
    if not value:
        # No models passed — default to all
        return list(REGISTRY.keys())

    # parse comma-separated
    parts = [m.strip() for m in value.split(",") if m.strip()]
    invalid = [m for m in parts if m not in REGISTRY]
    if invalid:
        valid = ", ".join(sorted(REGISTRY.keys()))
        raise click.BadParameter(f"Invalid model name(s): {invalid}. Valid models are: {valid}")
    return parts


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
    epilog=(
        "EXAMPLES:\n\n"
        "  python -m tools.run --models dtree,xgboost --n-trials 5 --cv-folds 4\n\n"
        "  python -m tools.run  # runs all models by default\n\n"
    ),
)
@click.option(
    "--models",
    callback=_validate_model_names,
    default=None,
    help=(
        "Comma-separated model names to run (e.g. 'lr,dtree,xgboost'). "
        "If omitted, all models in REGISTRY will be run.\n\nValid values: " + ", ".join(sorted(REGISTRY.keys()))
    ),
)
@click.option(
    "--data-path",
    default=None,
    help="Path to the dataset CSV file. Overrides default in settings.",
)
@click.option(
    "--n-trials",
    default=3,
    type=int,
    show_default=True,
    help="Number of Optuna trials per model.",
)
@click.option(
    "--cv-folds",
    default=3,
    type=int,
    show_default=True,
    help="Number of cross-validation folds.",
)
@click.option(
    "--scoring",
    default="neg_mean_squared_error",
    show_default=True,
    help="Scoring metric to optimize.",
)
@click.option(
    "--best-model-registry-name",
    default="ubereats-price-predictor",
    show_default=True,
    help="Name under which best model is registered in Mlflow Model Registry.",
)
def main(
    models,  # this is already a list[str] thanks to callback
    data_path,
    n_trials,
    cv_folds,
    scoring,
    best_model_registry_name,
):
    """
    Restaurant Menu Pricing Project CLI v0.0.1.

    Main entry point for the pipeline execution.
    This entrypoint is where everything comes together.

    Run the full training and hyperparameter tuning pipeline for selected models.
    """
    apply_global_settings()

    # Setup mlflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    # `models` is already a list of validated names
    logger.info(f"Running pipeline for models: {models}")

    data_path = data_path or settings.DATASET_SAMPLED_PATH

    result = autotune_pipeline(
        model_names=models,
        data_path=data_path,
        n_trials=n_trials,
        cv_folds=cv_folds,
        scoring=scoring,
        best_model_registry_name=best_model_registry_name,
    )
    logger.info(f"Best model: {result['best_model_name']}")


if __name__ == "__main__":
    main()
