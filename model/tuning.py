import optuna
import pandas as pd
from loguru import logger
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
    scoring_criterion: str = "neg_mean_absolute_error",
) -> dict[str, int]:
    """Optuna tuning for RandomForestRegressor (minimize MAE via 3-fold CV)."""

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


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    n_trials: int = 3,
    scoring_criterion: str = "neg_mean_absolute_error",
) -> dict[str, float | int]:
    """Optuna tuning for XGBRegressor (minimize MAE via 3-fold CV)."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = XGBRegressor(random_state=settings.SEED, **params)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        cv = KFold(n_splits=3, shuffle=True, random_state=settings.SEED)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring_criterion)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Best XGB params: {study.best_params}")
    return study.best_params, study.best_value  # type: ignore[return-value]
