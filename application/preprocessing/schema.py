from dataclasses import dataclass

from core import settings


@dataclass(frozen=True)
class ModelSchema:
    """
    Central definition of dataset schema used for model training and evaluation.

    Each attribute groups columns of a specific semantic type.
    Keeping them here ensures a single source of truth across
    data loading, preprocessing, and model training code.
    """

    numeric: tuple[str, ...] = ("cost_of_living_index", "density")
    categorical: tuple[str, ...] = ("category", "price_range", "state_id")
    text: tuple[str, ...] = ("ingredients",)
    target: str = getattr(settings, "TARGET", "price")  # Target variable for prediction
    # Column used for stratified splitting (or data split flag)
    data_split_col: str = settings.DATA_SPLIT_COL or "category"

    # -------------------------------------------------------------------------
    # Derived helpers
    # -------------------------------------------------------------------------

    def feature_cols(self) -> tuple[str, ...]:
        """Return all input feature columns (in deterministic order)."""
        return *self.numeric, *self.categorical, *self.text

    def expected_cols(self) -> tuple[str, ...]:
        """Return the full expected set of columns (features + target)."""
        return *self.feature_cols(), self.target

    # -------------------------------------------------------------------------
    # Validation utilities
    # -------------------------------------------------------------------------

    def validate(self, df) -> None:
        """
        Validate that the DataFrame contains all expected columns.
        Raises a ValueError if any are missing.
        """
        missing = set(self.expected_cols()) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")


# -------------------------------------------------------------------------
# Instantiate a default, reusable schema object
# -------------------------------------------------------------------------
schema = ModelSchema()
