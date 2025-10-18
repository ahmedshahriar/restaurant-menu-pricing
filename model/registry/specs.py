from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from core.settings import settings

from .spaces import dtree_space, lgbm_space, rf_space, xgb_space


@dataclass(frozen=True)
class ModelSpec:
    """
    one class to bind a regressor estimator to its Optuna space.
    - name: short string name for registry lookup
    - estimator_cls: the sklearn-compatible regressor class
    - base_kwargs: stable defaults you always want (n_jobs, random_state, etc.)
    - param_space: function that returns a dict of hyperparameters from a trial object
    """

    name: str
    estimator_cls: type[BaseEstimator]
    base_kwargs: dict[str, Any]
    param_space: Callable[[optuna.Trial], dict[str, Any]] | None

    def build(self, params: dict[str, Any] | None = None) -> BaseEstimator:
        """Instantiate estimator with base kwargs merged with tuned params."""
        # tuned params should override base defaults if keys collide
        params = params or {}
        merged = {**self.base_kwargs, **params}
        return self.estimator_cls(**merged)


# -------- registry entries (add more models by adding more entries) --------

LR_REG = ModelSpec(
    name="lr",
    estimator_cls=LinearRegression,
    base_kwargs={
        "n_jobs": -1,
    },
    param_space=None,
)

RIDGE_REG = ModelSpec(
    name="ridge",
    estimator_cls=Ridge,
    base_kwargs={
        "random_state": settings.SEED,
    },
    param_space=lambda trial: {
        "alpha": trial.suggest_float("alpha", 1e-5, 1e2, log=True),
    },
)

LASSO_REG = ModelSpec(
    name="lasso",
    estimator_cls=Lasso,
    base_kwargs={
        "random_state": settings.SEED,
    },
    param_space=lambda trial: {
        "alpha": trial.suggest_float("alpha", 1e-5, 1e2, log=True),
    },
)

DTREE_REG = ModelSpec(
    name="dtree",
    estimator_cls=DecisionTreeRegressor,
    base_kwargs={
        "random_state": settings.SEED,
    },
    param_space=dtree_space,
)


RF_REG = ModelSpec(
    name="rforest",
    estimator_cls=RandomForestRegressor,
    base_kwargs={
        "n_jobs": -1,
        "random_state": settings.SEED,
    },
    param_space=rf_space,
)

XGB_REG = ModelSpec(
    name="xgboost",
    estimator_cls=XGBRegressor,
    base_kwargs={
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": settings.SEED,
    },
    param_space=xgb_space,
)

LGBM_REG = ModelSpec(
    name="lightgbm",
    estimator_cls=LGBMRegressor,
    base_kwargs={
        "n_jobs": -1,
        "random_state": settings.SEED,
    },
    param_space=lgbm_space,
)

REGISTRY: dict[str, ModelSpec] = {
    LR_REG.name: LR_REG,
    RIDGE_REG.name: RIDGE_REG,
    LASSO_REG.name: LASSO_REG,
    RF_REG.name: RF_REG,
    DTREE_REG.name: DTREE_REG,
    XGB_REG.name: XGB_REG,
    LGBM_REG.name: LGBM_REG,
}


def get_model_spec(name: str) -> ModelSpec:
    """Fetch a ModelSpec from the registry by name, or raise ValueError if not found."""
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Options: {list(REGISTRY)}") from exc
