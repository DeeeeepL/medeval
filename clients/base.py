from abc import ABC, abstractmethod
from typing import List, Dict

class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], model: str | None = None,
             temperature: float | None = None) -> str:
        ...
