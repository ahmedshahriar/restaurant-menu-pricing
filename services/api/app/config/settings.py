from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    SCORING_URI: AnyHttpUrl | None = None
    REQUEST_TIMEOUT: float = 10.0
    AML_DEPLOYMENT: str | None = None  # e.g., "blue" or "green"


settings = Settings()
