"""CLI 公共工具：构建模型、攻击函数、防御函数。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# 触发插件注册
import attacks  # noqa: F401
import defenses  # noqa: F401

from core.config import load_settings, model_settings
from core.registry import get_attack_cls, get_defense_cls
from models import OpenAIModel, MockModel


def build_model(mock: bool | None = None):
    """构建模型实例。mock=None 时自动判断（无 api_key 则用 Mock）。"""
    settings = load_settings()
    mcfg = model_settings(settings)
    if mock is None:
        mock = not bool(mcfg.get("api_key"))
    if mock:
        print("[model] 使用 MockModel（无真实 LLM 调用，仅供冒烟演练）")
        return MockModel()
    print(f"[model] OpenAI 兼容: model={mcfg.get('model')} "
          f"base_url={mcfg.get('base_url') or 'https://api.openai.com/v1'}")
    return OpenAIModel.from_settings(mcfg)


def build_attack_fn(name: str) -> Callable:
    """返回 attack(env) -> str 函数。"""
    cls = get_attack_cls(name)
    inst = cls()

    def attack_fn(env):
        return inst.attack(env)

    attack_fn.__name__ = f"attack[{name}]"
    return attack_fn


def build_defense_fn(name: str) -> Callable:
    """返回 defend(env) -> str 函数。"""
    cls = get_defense_cls(name)
    inst = cls()

    def defend_fn(env):
        return inst.defend(env)

    defend_fn.__name__ = f"defend[{name}]"
    return defend_fn
