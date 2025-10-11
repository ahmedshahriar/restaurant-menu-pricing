from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # seed for reproducibility
    SEED: int = 33
    PYTHONHASHSEED: str = str(SEED)
    TORCH_NUM_THREADS: int = 2

    # ---- Visualization ----
    MPL_FIGSIZE: tuple[int, int] = (12, 4)
    MPL_DPI: int = 150

    # ---- Warnings ----
    IGNORE_DEPRECATION_WARNINGS: bool = True
    IGNORE_FUTURE_WARNINGS: bool = True

    MLFLOW_TRACKING_URI: str | None = None
    MLFLOW_EXPERIMENT_NAME: str | None = None

    # kaggle config
    KAGGLE_USERNAME: str | None = None
    KAGGLE_KEY: str | None = None

    # huggingface token
    HUGGINGFACE_ACCESS_TOKEN: str | None = None

    # MongoDB database
    DATABASE_HOST: str | None = None
    DATABASE_NAME: str | None = None
    DATABASE_COLLECTION: str | None = None

    # dataset paths
    DATASET_SAMPLED_PATH: str | None = None
    DATASET_SAMPLED_WITH_EMBEDDINGS_PATH: str | None = None

    # model training/tuning config
    TARGET: str | None = None
    DATA_SPLIT_COL: str | None = None
    TEST_SIZE: float | None = None
    N_TRIALS: int | None = None
    CV_FOLDS: int | None = None
    SCORING: str | None = None
    BEST_MODEL_REGISTRY_NAME: str | None = None

    # artifacts directory
    ARTIFACT_DIR: str | None = None


settings = Settings()
