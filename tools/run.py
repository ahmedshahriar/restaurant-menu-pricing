from __future__ import annotations

import click
import mlflow
from loguru import logger

from application.config import apply_global_settings, configure_mlflow_backend
from application.dataset import generate_training_sample
from core import __version__, settings
from model import REGISTRY
from pipelines import autotune_pipeline, dwh_export_pipeline

HELP_TEXT = f"""
Restaurant Menu Pricing CLI v{__version__}

Runs the complete data and machine learning pipeline for restaurant price modeling.

\b
This tool:
- Crawls Uber Eats (USA) for restaurant and menu data
- Stores raw and processed data in the data warehouse
- Generates sampled datasets for model training
- Preprocesses and transforms features
- Performs hyperparameter tuning and model comparison
- Registers and saves the best-performing model to AzureML/MLflow Model Registry

\b
Pipeline sequence:
ETL crawl → data warehouse → sample → preprocess → tune → ML models compare → register to AzureMl/MLFlow
"""

MODEL_CHOICES = sorted(REGISTRY.keys())


def _validate_model_names(_: click.Context, __: click.Option, value: str | None) -> list[str]:
    """
    click callback to validate the `--models` input.
    `value` is a comma-separated string (or None).
    Returns a list of model names.
    """
    if not value:
        # No models passed — default to all
        return list(MODEL_CHOICES)

    # parse comma-separated
    parts = [m.strip() for m in value.split(",") if m.strip()]
    invalid = [m.strip() for m in parts if m not in REGISTRY]
    if invalid:
        valid = ", ".join(sorted(MODEL_CHOICES))
        raise click.BadParameter(f"Invalid model name(s): {invalid}. Valid models are: {valid}")
    # require at least two models
    if len(parts) < 2:
        raise click.BadParameter(
            "Please provide at least two models for comparison (e.g. poetry poe run-models lr,dtree)."
        )

    return parts


def _print_plan(models, data_path, n_trials, cv_folds, scoring, best_model_registry_name):
    click.echo(
        "Plan:\n"
        f"  Models: {models}\n"
        f"  Data path: {data_path or '<settings default>'}\n"
        f"  Optuna trials: {n_trials}, CV folds: {cv_folds}\n"
        f"  Scoring criterion: {scoring}\n"
        f"  Best model registry name: {best_model_registry_name}\n"
    )


@click.group(
    name="restaurant-menu-pricing-cli",
    invoke_without_command=True,
    help=HELP_TEXT,
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=False,
    epilog=(
        "EXAMPLES:\n\n"
        "python -m tools.run  # runs all models by default\n\n"
        "python -m tools.run --list-models  # list available models\n\n"
        "python -m tools.run --dry-run  # show the plan without running\n\n"
        "python -m tools.run --models dtree,xgboost --n-trials 5 --cv-folds 4\n\n"
    ),
)
@click.version_option(
    version=__version__, message="Restaurant Menu Pricing CLI v%(version)s", prog_name="Restaurant CLI"
)
@click.option(
    "--list-models",
    is_flag=True,
    help="List available model names and exit.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the resolved plan (models/options) and exit without running.",
)
@click.option(
    "--models",
    callback=_validate_model_names,
    default=None,
    help=(
        "Comma-separated model names to run (e.g. 'lr,dtree,xgboost'). "
        "If omitted, all models in REGISTRY will be run. "
        "\n\nValid values: " + ", ".join(sorted(MODEL_CHOICES))
    ),
)
@click.option(
    "--sampled-data-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=str),
    default=None,
    envvar="DATA_PATH",
    help="Path to the dataset CSV file (can also be set via DATA_PATH). Overrides the default in settings.",
)
@click.option(
    "--n-trials",
    type=int,
    show_default=True,
    envvar="N_TRIALS",
    help="Number of Optuna trials per model.",
)
@click.option(
    "--cv-folds",
    type=int,
    show_default=True,
    envvar="CV_FOLDS",
    help="Number of cross-validation folds.",
)
@click.option(
    "--scoring",
    show_default=True,
    envvar="SCORING",
    help="Scoring metric to optimize.",
)
@click.option(
    "--best-model-registry-name",
    show_default=True,
    envvar="BEST_MODEL_REGISTRY_NAME",
    help="Name under which best model is registered in Mlflow Model Registry.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    models: list[str],
    list_models: bool,
    sampled_data_path: str | None,
    n_trials: int,
    cv_folds: int,
    scoring: str,
    best_model_registry_name: str,
    dry_run: bool,
) -> None:
    sampled_data_path = sampled_data_path or settings.SAMPLED_DATA_PATH
    n_trials = n_trials or settings.N_TRIALS
    cv_folds = cv_folds or settings.CV_FOLDS
    scoring = scoring or settings.SCORING
    best_model_registry_name = best_model_registry_name or settings.BEST_MODEL_REGISTRY_NAME

    # If a subcommand is used, don't run default action
    if ctx.invoked_subcommand:
        return

    # dry-run: just show the plan and exit
    if dry_run:
        _print_plan(models, sampled_data_path, n_trials, cv_folds, scoring, best_model_registry_name)
        raise SystemExit(0)
    # quick list-and-exit
    elif list_models:
        click.echo("Available models:\n  " + "\n  ".join(sorted(MODEL_CHOICES)))
        raise SystemExit(0)
    else:
        # apply global settings (seed, matplotlib, warnings)
        apply_global_settings()

        _print_plan(models, sampled_data_path, n_trials, cv_folds, scoring, best_model_registry_name)

        # Setup mlflow
        tracking_uri = configure_mlflow_backend()
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        logger.info(f"MLflow configured. Tracking URI -> {tracking_uri}")

        # `models` is already a list of validated names
        logger.info(f"Running pipeline for models: {models}")

        try:
            result = autotune_pipeline(
                model_names=models,
                data_path=sampled_data_path,
                n_trials=n_trials,
                cv_folds=cv_folds,
                scoring=scoring,
                best_model_registry_name=best_model_registry_name,
            )
            logger.info(f"Best model: {result['best_model_name']}")
        except Exception as e:
            # error and non-zero exit
            raise click.ClickException(str(e)) from e


# --------------------------
# New: simple no-arg generator command
# --------------------------
@cli.command("generate-train-sample")
def generate():
    """
    Generates a sampled, feature-enriched training dataset from the published data warehouse exports on Kaggle.

    \b
    - Output: {settings.TRAINING_DATA_SAMPLE_PATH}
    - Produces a cleaned, enriched subset of crawled restaurant data for model training.
    - Includes NER-extracted ingredients, cost-of-living index, and location features (e.g., population density).
    - Removes price outliers and normalizes price ranges into buckets.
    - Requires internet access for NER model download on first run.

    Sampling and filtering logic (to be made configurable):
    - Focuses on top restaurant categories (e.g., Sandwiches, Salads, Wraps).
    - Limits to top categories (e.g., 15) per city and top cities (e.g., 5) per state.
    - Filters for top US states by restaurant count (e.g., TX, VA, WA, WI, UT).

    """
    try:
        # apply global settings (seed, matplotlib, warnings)
        apply_global_settings()
        _ = generate_training_sample()
        logger.info(f"Data generation complete -> {settings.SAMPLED_DATA_PATH}")
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        raise click.ClickException(str(e)) from e


@cli.command("dwh-export")
def dwh_export():
    """
    Exports raw restaurant and menu data from the data warehouse (MongoDB) to normalized CSV files for
    downstream publishing/consumption.

    \b
    - Output directory: {settings.DWH_EXPORT_DIR}
    - Restaurant data file: {settings.RESTAURANT_DATA_PATH}
    - Menu data file: {settings.MENU_DATA_PATH}
    - Requires MongoDB connection settings to be configured in environment variables or .env file.
    - Example env vars:
        - DATABASE_HOST
        - DATABASE_NAME
        - DATABASE_COLLECTION

    """
    try:
        # apply global settings (seed, matplotlib, warnings)
        apply_global_settings()
        logger.info("Starting MongoDB export job...")
        dwh_export_pipeline()
        logger.info(f"DWH export job completed successfully -> {settings.DWH_EXPORT_DIR}")
    except Exception as e:
        logger.error(f"DWH export job failed: {e}")
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    cli()
