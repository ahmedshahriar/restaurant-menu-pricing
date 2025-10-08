from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

from .schema import CATEGORICAL_COLS, NUMERIC_COLS, TEXT_COLS
from .text import dummy as _dummy  # or inline if tiny


def build_preprocessor() -> ColumnTransformer:
    num_tf = Pipeline([("scaler", MinMaxScaler())])
    cat_tf = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    text_tf = Pipeline(
        [
            ("squeeze", FunctionTransformer(lambda x: x.squeeze())),
            ("tfidf", TfidfVectorizer(analyzer="word", tokenizer=_dummy, preprocessor=_dummy, token_pattern=None)),
        ]
    )
    return ColumnTransformer(
        [("num", num_tf, NUMERIC_COLS), ("cat", cat_tf, CATEGORICAL_COLS), ("text", text_tf, TEXT_COLS)],
        n_jobs=-1,
    )
