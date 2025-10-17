from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

from .schema import schema


def build_preprocessor() -> ColumnTransformer:
    num_tf = Pipeline([("scaler", MinMaxScaler())])
    cat_tf = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    text_tf = Pipeline(
        [
            # Always return a Series (1 row => length-1 Series), no squeeze()
            # best approach: preserve list-of-phrases as-is, no join (e.g., "chopped onions" stays intact)
            ("pick_col", FunctionTransformer(lambda x: x.iloc[:, 0], validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    # Inline identity functions (avoid external imports like a dummy function)
                    # Example:
                    #     def dummy(doc): return doc
                    # This avoids external module dependencies (e.g., `application.text`)
                    # and ensures the model can be unpickled and served without missing imports.
                    tokenizer=lambda x: x,
                    preprocessor=lambda x: x,
                    token_pattern=None,  # required when you provide your own tokenizer
                    lowercase=False,  # optional: keep original casing if important
                    ngram_range=(1, 2),  # unigrams + bigrams
                    min_df=2,  # ignore very rare phrases
                ),
            ),
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
