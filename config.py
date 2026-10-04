import os
from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4o-mini")

class Settings(BaseSettings):
    # DATABASE_URL: str
    # TEST_DATABASE_URL: str | None = None

    APP_ENV: str = "development"

    LOG_LEVEL: str = "INFO"

    # File logging
    LOG_TO_FILE: bool = False
    LOG_FILE: str = "logs/app.log"
    LOG_MAX_BYTES: int = 10_000_000  # 10 MB
    LOG_BACKUP_COUNT: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()