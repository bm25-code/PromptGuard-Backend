"""配置加载：合并 settings.yaml、任务 yaml 与环境变量。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 项目根目录（PromptGuard-Backend/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
TASKS_DIR = PROJECT_ROOT / "config" / "tasks"


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base，返回新 dict。"""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings() -> dict:
    """加载主配置 settings.yaml。"""
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_task_config(dataset: str) -> dict:
    """加载 config/tasks/<dataset>.yaml。"""
    path = TASKS_DIR / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"任务配置不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_datasets() -> list[str]:
    """列出所有可用任务（按 yaml 文件名）。"""
    return sorted(p.stem for p in TASKS_DIR.glob("*.yaml"))


def resolve_path(p: str) -> Path:
    """将相对路径解析为相对项目根的绝对路径。"""
    pp = Path(p)
    if not pp.is_absolute():
        pp = PROJECT_ROOT / pp
    return pp


def model_settings(settings: dict | None = None) -> dict[str, Any]:
    """合并 settings.yaml 的 model 段与 OPENAI_* 环境变量，环境变量优先。"""
    settings = settings or load_settings()
    m = dict(settings.get("model", {}))
    if os.environ.get("OPENAI_API_KEY"):
        m["api_key"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_BASE_URL"):
        m["base_url"] = os.environ["OPENAI_BASE_URL"]
    if os.environ.get("OPENAI_MODEL"):
        m["model"] = os.environ["OPENAI_MODEL"]
    if os.environ.get("OPENAI_TEMPERATURE"):
        m["temperature"] = float(os.environ["OPENAI_TEMPERATURE"])
    if os.environ.get("OPENAI_MAX_TOKENS"):
        m["max_tokens"] = int(os.environ["OPENAI_MAX_TOKENS"])
    return m
