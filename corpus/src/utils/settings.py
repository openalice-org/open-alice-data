"""Environment-backed settings, loaded once from the process env and `.env`."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    openalex_api_key: str = ""
    openalex_mailto: str = ""


settings = Settings()
