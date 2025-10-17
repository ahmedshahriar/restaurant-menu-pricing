from __future__ import annotations

from typing import Any

import optuna


# -------------------- XGBoost (Regressor) --------------------
def xgb_space(trial: optuna.Trial) -> dict[str, Any]:
    """Compact XGBRegressor search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


# -------------------- LightGBM (Regressor) --------------------
def lgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    """Compact LGBMRegressor search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1500, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", -1, 15),  # -1 means no limit
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),  # same as bagging_fraction
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
    }


def dtree_space(trial: optuna.Trial) -> dict[str, Any]:
    """Compact DecisionTreeRegressor search space."""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


# ---------------- Random Forest (Regressor) -----------------
def rf_space(trial: optuna.Trial) -> dict[str, Any]:
    """Compact RandomForestRegressor search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
