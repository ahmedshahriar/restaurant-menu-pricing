import numpy as np
import pandas as pd
import pytest


def test_splitter_stratified_and_shapes(monkeypatch):
    sp = pytest.importorskip("application.dataset.io.splitter", reason="splitter module not found")

    # Ensure the module-level constants are set even if import order differed in CI
    monkeypatch.setattr(sp, "DATA_SPLIT_COL", "category", raising=False)
    monkeypatch.setattr(sp, "TARGET_COL", "price", raising=False)

    df = pd.DataFrame(
        {
            "category": ["A"] * 8 + ["B"] * 2,
            "price": np.arange(10, dtype=float),
            "feat1": range(10),
        }
    )

    assert hasattr(sp, "split_data"), "split_data() not exported from splitter.py"
    X_train, X_test, y_train, y_test = sp.split_data(df, test_size=0.2)

    # sizes add up
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)

    # target NOT in X’s columns
    assert "price" not in X_train.columns and "price" not in X_test.columns

    # basic stratification check
    assert set(X_train["category"]).issubset({"A", "B"})
    assert set(X_test["category"]).issubset({"A", "B"})
