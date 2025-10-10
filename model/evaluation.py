import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from core.settings import settings


def evaluate_model(n_folds: int, model: Pipeline, X: pd.DataFrame, y: pd.Series, scoring_criterion: str) -> np.ndarray:
    # TODO: Add support for additional scoring options such as 'f1', 'roc_auc', 'precision', and 'recall'.
    """Evaluate a model using K-Fold cross-validation and return the scores."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=settings.SEED)
    scores = cross_val_score(
        model,
        X,
        y,
        scoring=scoring_criterion,
        cv=kf,
        n_jobs=-1,
        error_score="raise",
    )
    return scores
