import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit

from application.preprocessing import DATA_SPLIT_COL, TARGET_COL
from core.settings import settings


def split_data(
    df: pd.DataFrame, test_size: float = settings.TEST_SIZE if settings.TEST_SIZE else 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into stratified train and test sets based on the 'category' column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and 'price' target.
    test_size : float, optional
        Fraction of data to reserve for testing, by default 0.2.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    strat_train_set, strat_test_set = None, None
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=settings.SEED)

    for train_idx, test_idx in strat_split.split(df, df[DATA_SPLIT_COL]):
        strat_train_set = df.loc[train_idx].copy()
        strat_test_set = df.loc[test_idx].copy()

    X_train = strat_train_set.drop(columns=[TARGET_COL])
    y_train = strat_train_set[TARGET_COL]

    X_test = strat_test_set.drop(columns=[TARGET_COL])
    y_test = strat_test_set[TARGET_COL]

    # Log split shapes
    logger.info(f"Data split: {len(df)} rows, train:test = {1 - test_size:.0f}:{test_size:.0f}")
    logger.info(f"Shape --> X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"Shape --> X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Log stratification distribution
    logger.info(f"Train category distribution:\n{strat_train_set['category'].value_counts(normalize=True).round(2)}")
    logger.info(f"Test category distribution:\n{strat_test_set['category'].value_counts(normalize=True).round(2)}")

    return X_train, X_test, y_train, y_test
