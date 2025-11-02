from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # mlflow config
    MLFLOW_BACKEND: str | None = None  # local | azure
    MLFLOW_TRACKING_URI: str | None = None
    MLFLOW_EXPERIMENT_NAME: str | None = None

    # --- Azure workspace IDs (used when MLFLOW_BACKEND=azure) ---
    AZURE_SUBSCRIPTION_ID: str | None = None
    AZURE_RESOURCE_GROUP: str | None = None
    AZURE_ML_WORKSPACE_NAME: str | None = None
    AZURE_UAMI_NAME: str | None = None

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

    # kaggle config
    KAGGLE_USERNAME: str | None = None
    KAGGLE_KEY: str | None = None

    # huggingface token
    HUGGINGFACE_ACCESS_TOKEN: str | None = None

    CRAWLED_TASK_DATA_PATH: str | None = None
    # Proxy Config for web crawling
    PROXY_HOST: str | None = None
    PROXY_PORT: int | None = None
    PROXY_USER: str | None = None
    PROXY_PASSWORD: str | None = None

    # MongoDB database (DWH)
    # alternative to S3 / Azure Blob Storage
    DATABASE_HOST: str | None = None
    DATABASE_NAME: str | None = None
    DATABASE_COLLECTION: str | None = None

    # data warehouse export directory
    DWH_EXPORT_DIR: str | None = None
    RESTAURANT_DATA_PATH: str | None = None
    MENU_DATA_PATH: str | None = None

    # dataset paths
    # generated featured dataset path
    # after feature engineering and cleaning
    SAMPLED_DATA_PATH: str | None = None
    SAMPLED_DATA_WITH_EMBEDDINGS_PATH: str | None = None

    # kaggle datasets
    # cost of living index by city
    INDEX_DS: str | None = None
    INDEX_FILE: str | None = None

    # us cities database with lat/long info
    DENSITY_DS: str | None = None
    DENSITY_FILE: str | None = None

    # used to build states_name_dict
    STATES_DS: str | None = None
    STATES_FILE: str | None = None

    # artifacts directory
    ARTIFACT_DIR: str | None = None

    # Food NER Model
    NER_MODEL: str | None = None

    # model training/tuning config
    TARGET: str | None = None
    DATA_SPLIT_COL: str | None = None
    TEST_SIZE: float | None = None
    N_TRIALS: int | None = None
    CV_FOLDS: int | None = None
    SCORING: str | None = None

    # batch size for inference time during training/tuning
    # configurable; controls timing sample size
    BATCH_SIZE_INFER_TEST: int | None = None

    # final best model registry name
    # if set, the best model is registered under this name in MLflow Model Registry
    BEST_MODEL_REGISTRY_NAME: str | None = None
    MODEL_ENDPOINT_NAME: str | None = None

    # model serving
    MODEL_SERVE_PORT: int = 5000


settings = Settings()
