# 📁 ai_module/providers/llm/google.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from .base import BaseLLMProvider

class GoogleLLMProvider(BaseLLMProvider):
    """Google LLM provider for Gemini models."""

    def __init__(self, model_name: str, api_key: str):
        if not api_key:
            raise ValueError("Google API key must be provided.")
        
        self.model_name = model_name
        self.api_key = SecretStr(api_key)

    def get_llm(self, temperature: float = 0.0) -> BaseChatModel:
        """Initializes and returns the ChatGoogleGenerativeAI instance."""
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key.get_secret_value(),
            temperature=temperature
        )