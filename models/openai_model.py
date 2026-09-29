"""OpenAI 兼容模型适配器。

通过标准 openai SDK 访问，base_url / api_key / model 可配置，
兼容官方 OpenAI、Azure OpenAI（兼容网关）、本地 vLLM / Ollama / LMDeploy 等。

模型只需实现 query(prompt) -> str 接口即可被引擎使用。
"""
from __future__ import annotations

import time
from typing import Any, Optional

from openai import OpenAI
import openai


class OpenAIModel:
    """标准 OpenAI Chat Completions 接口封装。

    query(prompt) 将整段 prompt 作为单条 user 消息发送（与赛题基线一致）。
    """

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.1,
        max_tokens: int = 150,
        seed: Optional[int] = 100,
        max_retries: int = 3,
        retry_sleep: float = 5.0,
    ):
        if not api_key:
            raise ValueError(
                "缺少 OPENAI_API_KEY。请在 .env 或 config/settings.yaml 中配置。"
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    @classmethod
    def from_settings(cls, model_cfg: dict) -> "OpenAIModel":
        """从 model_settings() 字典构造。"""
        return cls(
            model=model_cfg.get("model", "gpt-4o-mini"),
            api_key=model_cfg.get("api_key", ""),
            base_url=model_cfg.get("base_url", ""),
            temperature=float(model_cfg.get("temperature", 0.1)),
            max_tokens=int(model_cfg.get("max_tokens", 150)),
            seed=model_cfg.get("seed", 100),
            max_retries=int(model_cfg.get("max_retries", 3)),
            retry_sleep=float(model_cfg.get("retry_sleep", 5.0)),
        )

    def query(self, prompt: str) -> str:
        """单条 user 消息请求，带重试。返回模型响应文本。"""
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._do_query(prompt)
            except openai.RateLimitError as e:
                last_err = e
                time.sleep(self.retry_sleep)
            except openai.APIConnectionError as e:
                last_err = e
                time.sleep(self.retry_sleep)
            except openai.BadRequestError as e:
                # 请求格式错误，重试无意义
                return f"[BadRequestError] {e}"
            except openai.APIStatusError as e:
                last_err = e
                # 5xx 重试，4xx 直接返回
                if 500 <= e.status_code < 600:
                    time.sleep(self.retry_sleep)
                else:
                    return f"[APIStatusError {e.status_code}] {e}"
            except Exception as e:
                last_err = e
                time.sleep(self.retry_sleep)
        return f"[QueryFailed after {self.max_retries} retries] {last_err}"

    def _do_query(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OpenAIModel(model={self.model!r}, "
            f"base_url={self.client.base_url!r})"
        )


class MockModel:
    """无网络/无 API Key 时的模拟模型，用于测试与冒烟演练。

    规则：若 prompt 中包含 attack_token 则原样回显 token（模拟被劫持）；
    否则按简单关键词给出任务答案。可通过注入 __mock_rules 自定义。
    """

    def __init__(self, rules: Optional[dict] = None):
        self.rules = rules or {}

    def query(self, prompt: str) -> str:
        # 模拟"被注入劫持"：prompt 中若出现形如 "output ... <token>" 的指令则回显 token
        import re
        # 抓取 prompt 中出现的最后一个全大写-数字 token（粗略模拟攻击成功）
        m = re.findall(r"\b([A-Z0-9]{4}-[A-Z0-9]{4}-\d{4})\b", prompt)
        # 仅当 prompt 明确要求输出该 token 时才回显
        for tok in reversed(m):
            if any(kw in prompt.lower() for kw in [
                "ignore", "output", "repeat", "print", "回复", "输出", "忽略",
            ]):
                return tok
        # 否则尝试完成原任务（简单关键词匹配）
        lower = prompt.lower()
        if "positive" in lower and "negative" in lower:
            return "positive" if "beautifully" in lower or "moving" in lower else "negative"
        if "entailment" in lower:
            return "entailment"
        if "equivalent" in lower:
            return "equivalent"
        if "spam" in lower:
            return "not spam"
        if "hateful" in lower or "offensive" in lower:
            return "no"
        return "ok"
