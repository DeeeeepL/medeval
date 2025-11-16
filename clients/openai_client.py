import requests
from typing import List, Dict, Optional
from .base import LLMClient

class OpenAIClient(LLMClient):
    def __init__(self, api_base: str, api_key: str,
                 default_model: str = "gpt-4o",
                 temperature: float = 0.0,
                 timeout: int = 120):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]],
             model: Optional[str] = None,
             temperature: Optional[float] = None) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
