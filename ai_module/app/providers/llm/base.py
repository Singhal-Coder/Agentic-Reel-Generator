# 📁 ai_module/providers/llm/base.py
from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_llm(self, temperature: float = 0.0) -> BaseChatModel:
        """Returns a LangChain compatible chat model instance."""
        pass