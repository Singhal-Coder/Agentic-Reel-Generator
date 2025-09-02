# 📁 ai_module/config/settings.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal
from dotenv import load_dotenv


env_file_path = Path(__file__).resolve().parent.parent.parent / ".env.app"

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # --- Core Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
    CACHE_DIR: Path = BASE_DIR / "cache"
    OUTPUT_DIR: Path = BASE_DIR / "output"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Provider Selection ---
    LLM_PROVIDER: Literal["google", "openai"] = "google"

    # --- AI Models & API Keys ---
    GOOGLE_API_KEY: Optional[str] = None
    HUGGINGFACEHUB_ACCESS_TOKEN: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    PEXEL_API_KEY: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None
    GIPHY_API_KEY: Optional[str] = None
    MUSIC_API_KEY: Optional[str] = None
    
    # Model Names
    GOOGLE_LLM_NAME: str = "gemini-1.5-flash-latest"
    ELEVENLABS_MODEL_NAME: str = "eleven_flash_v2_5"

    MUSIC_API_BASE_URL: Optional[str] = None
    
    # --- LangSmith Tracing ---
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: Optional[str] = None
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    
    model_config = SettingsConfigDict(
        env_file=env_file_path,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore'
    )

settings = Settings()
load_dotenv(env_file_path)

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        settings.LOG_DIR,
        settings.CACHE_DIR,
        settings.OUTPUT_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

create_directories()