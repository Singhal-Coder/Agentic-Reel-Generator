from ..schemas.provider import Providers

from ..core.factories import get_llm_provider




providers = Providers(
    llm=get_llm_provider(0)
)