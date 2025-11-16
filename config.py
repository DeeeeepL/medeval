from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    api_base: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout: int = 120

@dataclass
class EvalConfig:
    """整体评测配置，包含待测模型和裁判模型两套配置。"""
    test: ModelConfig
    judge: ModelConfig


def load_eval_config() -> EvalConfig:
    """
    从环境变量加载：
    - TEST_API_BASE / TEST_API_KEY / TEST_MODEL
    - JUDGE_API_BASE / JUDGE_API_KEY / JUDGE_MODEL
    如果没单独配，就回落到 OPENAI_API_BASE / OPENAI_API_KEY。
    """
    common_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    common_key  = os.getenv("OPENAI_API_KEY", "")

    test_cfg = ModelConfig(
        api_base=os.getenv("TEST_API_BASE", common_base),
        api_key=os.getenv("TEST_API_KEY",  common_key),
        model=os.getenv("TEST_MODEL", "gpt-5.1"),
        temperature=float(os.getenv("TEST_TEMPERATURE", "0.0")),
        timeout=int(os.getenv("TEST_TIMEOUT", "120")),
    )

    judge_cfg = ModelConfig(
        api_base=os.getenv("JUDGE_API_BASE", common_base),
        api_key=os.getenv("JUDGE_API_KEY",  common_key),
        model=os.getenv("JUDGE_MODEL", "gpt-4o"),
        temperature=float(os.getenv("JUDGE_TEMPERATURE", "0.0")),
        timeout=int(os.getenv("JUDGE_TIMEOUT", "120")),
    )

    return EvalConfig(test=test_cfg, judge=judge_cfg)
