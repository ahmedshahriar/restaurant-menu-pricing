import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from core.settings import settings


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=settings.SEED).get_n_splits(X.values)
    scores = cross_val_score(
        model,
        X,
        y,
        scoring="neg_mean_squared_error",
        cv=kf,
        n_jobs=-1,
        error_score="raise",
    )
    return scores
