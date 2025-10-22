import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def test_model_schema_feature_and_expected_cols_are_consistent():
    from application.preprocessing.schema import schema

    # feature columns are the union of numeric + categorical + text, in order
    expected_features = tuple(schema.numeric) + tuple(schema.categorical) + tuple(schema.text)
    assert schema.feature_cols() == expected_features

    # expected_cols = features + target
    assert schema.expected_cols() == expected_features + (schema.target,)


def test_model_schema_validate_passes_and_raises():
    from application.preprocessing.schema import schema

    # Build a frame that has *all* required columns
    complete_df = pd.DataFrame(columns=list(schema.feature_cols()) + [schema.target])
    # Should not raise
    schema.validate(complete_df)

    # Now drop a few to exercise the error path
    missing_df = pd.DataFrame(columns=[schema.target] + list(schema.numeric))
    with pytest.raises(ValueError) as err:
        schema.validate(missing_df)

    msg = str(err.value)
    # Must mention at least one known missing column
    assert "Missing columns" in msg
    assert "category" in msg or "price_range" in msg or "state_id" in msg
