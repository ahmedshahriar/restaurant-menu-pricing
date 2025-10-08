from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

from .schema import schema
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
        [
            ("num", num_tf, list(schema.numeric)),
            ("cat", cat_tf, list(schema.categorical)),
            ("text", text_tf, list(schema.text)),
        ],
        n_jobs=-1,
    )
