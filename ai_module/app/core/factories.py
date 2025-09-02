# 📁 ai_module/core/factories.py
from langchain_core.language_models import BaseChatModel

from ..config.settings import settings

# --- LLM Provider Factory ---
def get_llm_provider(temp:float = 0.2) -> BaseChatModel:
    """Factory to create and return an LLM provider based on settings."""
    if settings.LLM_PROVIDER == "google":
        from ..providers.llm.google import GoogleLLMProvider
        return GoogleLLMProvider(settings.GOOGLE_LLM_NAME, settings.GOOGLE_API_KEY).get_llm(temp)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")