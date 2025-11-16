from dataclasses import dataclass
import os

@dataclass
class OpenAIConfig:
    api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    test_model: str = "gpt-5.1"   # 待测模型
    judge_model: str = "gpt-4o"   # 裁判模型
    temperature: float = 0.0
    timeout: int = 120

@dataclass
class RunConfig:
    seed: int = 42
    max_examples: int | None = None
    trials: int = 1                 # 选择题多次打乱
    nota_ratio: float = 0.0         # NOTA 干扰比例
    paraphrase_ratio: float = 0.0   # 问法扰动比例
    use_llm_judge: bool = True      # open_response 是否用 gpt-4o 裁判
