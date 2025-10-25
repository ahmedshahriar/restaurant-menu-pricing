import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytestmark = pytest.mark.unit


def _toy_frame_for_vectorizer():
    """
    Build a tiny DataFrame that matches the schema in schema.py.
    Ensure TF-IDF has tokens with min_df >= 2 so we don't hit empty vocabulary.
    """
    rows = [
        {
            "cost_of_living_index": 92.0,
            "density": 1156.0,
            "category": "Salads",
            "price_range": "cheap",
            "state_id": "wi",
            # tokens: onion, tomato appear in both rows -> pass min_df=2
            "ingredients": ["onion", "tomato", "lettuce"],
            "price": 9.99,
        },
        {
            "cost_of_living_index": 145.0,
            "density": 4300.0,
            "category": "Sandwiches",
            "price_range": "moderate",
            "state_id": "ca",
            "ingredients": ["tomato", "onion", "bread"],
            "price": 12.49,
        },
    ]
    return pd.DataFrame(rows)


def test_build_preprocessor_returns_fittable_column_transformer():
    from sklearn.compose import ColumnTransformer

    from application.preprocessing.schema import schema
    from application.preprocessing.transformers import build_preprocessor

    df = _toy_frame_for_vectorizer()

    ct = build_preprocessor()
    assert isinstance(ct, ColumnTransformer)

    # Fit/transform should succeed and return a sparse matrix
    X = ct.fit_transform(df[list(schema.feature_cols())])
    assert sparse.issparse(X) or isinstance(X, np.ndarray)
    # Should have 2 rows (same as input)
    assert X.shape[0] == len(df)

    # The transformer should include named blocks for num/cat/text using schema’s columns
    names = [name for name, _, _ in ct.transformers]
    assert {"num", "cat", "text"}.issubset(set(names))

    # Verify the column selections wired into the ColumnTransformer
    mapping = {name: cols for name, _, cols in ct.transformers}
    assert list(mapping["num"]) == list(schema.numeric)
    assert list(mapping["cat"]) == list(schema.categorical)
    assert list(mapping["text"]) == list(schema.text)


def test_text_pipeline_handles_small_corpus_gracefully():
    import numpy as np
    import pandas as pd
    from scipy import sparse

    from application.preprocessing.schema import schema
    from application.preprocessing.transformers import build_preprocessor

    # Two documents so min_df=2 is satisfiable
    df = pd.DataFrame(
        [
            {
                "cost_of_living_index": 100.0,
                "density": 2000.0,
                "category": "Wraps",
                "price_range": "moderate",
                "state_id": "tx",
                "ingredients": ["pepper", "salt", "pepper"],  # tokens appear in both docs
                "price": 8.5,
            },
            {
                "cost_of_living_index": 95.0,
                "density": 1800.0,
                "category": "Wraps",
                "price_range": "moderate",
                "state_id": "tx",
                "ingredients": ["salt", "pepper", "salt"],
                "price": 9.1,
            },
        ]
    )

    ct = build_preprocessor()
    X = ct.fit_transform(df[list(schema.feature_cols())])

    # basic shape/typing checks
    assert X.shape[0] == len(df)
    assert sparse.issparse(X) or isinstance(X, np.ndarray)
