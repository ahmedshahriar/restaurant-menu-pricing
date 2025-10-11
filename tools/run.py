from __future__ import annotations

import click
import mlflow
from loguru import logger

from application.config import apply_global_settings
from core import __version__, settings
from model import REGISTRY
from pipelines import autotune_pipeline

HELP_TEXT = f"""
    Restaurant Menu Pricing Project CLI v{__version__}.

    Main entry point for the pipeline execution.
    This entrypoint is where everything comes together.

    Run the full training and hyperparameter tuning pipeline for selected models.

    \b
    load -> split -> preprocess -> tune -> compare models -> register best model.
    """


def _validate_model_names(_: click.Context, __: click.Option, value: str | None) -> list[str]:
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


def _print_plan(models, data_path, n_trials, cv_folds, scoring, best_model_registry_name):
    click.echo(
        "Plan:\n"
        f"  Models: {models}\n"
        f"  Data path: {data_path or '<settings default>'}\n"
        f"  Optuna trials: {n_trials}, CV folds: {cv_folds}\n"
        f"  Scoring criterion: {scoring}\n"
        f"  Best model registry name: {best_model_registry_name}\n"
    )


@click.command(
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
    "--models",
    callback=_validate_model_names,
    default=None,
    help=(
        "Comma-separated model names to run (e.g. 'lr,dtree,xgboost'). "
        "If omitted, all models in REGISTRY will be run. "
        "\n\nValid values: " + ", ".join(sorted(REGISTRY.keys()))
    ),
)
@click.option(
    "--list-models",
    is_flag=True,
    help="List available model names and exit.",
)
@click.option(
    "--data-path",
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
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the resolved plan (models/options) and exit without running.",
)
def main(
    models: list[str],
    list_models: bool,
    data_path: str | None,
    n_trials: int,
    cv_folds: int,
    scoring: str,
    best_model_registry_name: str,
    dry_run: bool,
) -> None:
    # apply global settings (seed, matplotlib, warnings)
    apply_global_settings()

    # Setup mlflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    data_path = data_path or settings.DATASET_SAMPLED_PATH
    n_trials = n_trials or settings.N_TRIALS
    cv_folds = cv_folds or settings.CV_FOLDS
    scoring = scoring or settings.SCORING
    best_model_registry_name = best_model_registry_name or settings.BEST_MODEL_REGISTRY_NAME

    # quick list-and-exit
    if list_models:
        click.echo("Available models:\n  " + "\n  ".join(sorted(REGISTRY.keys())))
        raise SystemExit(0)

    # dry-run: just show the plan and exit
    if dry_run:
        _print_plan(models, data_path, n_trials, cv_folds, scoring, best_model_registry_name)
        raise SystemExit(0)

    # `models` is already a list of validated names
    logger.info(f"Running pipeline for models: {models}")

    try:
        _print_plan(models, data_path, n_trials, cv_folds, scoring, best_model_registry_name)
        result = autotune_pipeline(
            model_names=models,
            data_path=data_path,
            n_trials=n_trials,
            cv_folds=cv_folds,
            scoring=scoring,
            best_model_registry_name=best_model_registry_name,
        )
        logger.info(f"Best model: {result['best_model_name']}")
    except Exception as e:
        # error and non-zero exit
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    main()
