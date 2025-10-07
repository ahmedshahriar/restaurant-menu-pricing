from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # seed for reproducibility
    SEED: int = 42
    PYTHONHASHSEED: str = str(SEED)

    # kaggle config
    KAGGLE_USERNAME: str | None = None
    KAGGLE_KEY: str | None = None

    # MongoDB database
    DATABASE_HOST: str | None = None
    DATABASE_NAME: str | None = None
    DATABASE_COLLECTION: str | None = None

    # huggingface token
    HUGGINGFACE_ACCESS_TOKEN: str | None = None


settings = Settings()
