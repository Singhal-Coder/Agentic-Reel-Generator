from pydantic import BaseModel, ConfigDict

from langchain_core.language_models import BaseChatModel

class Providers(BaseModel):
    llm: BaseChatModel
    model_config = ConfigDict(arbitrary_types_allowed=True)