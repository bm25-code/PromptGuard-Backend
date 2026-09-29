"""样本加载工具。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import load_settings, resolve_path
from .sample import Sample


def samples_path(dataset: str, settings: Optional[dict] = None) -> Path:
    settings = settings or load_settings()
    return resolve_path(settings["paths"]["samples_dir"]) / f"{dataset}.json"


def load_samples(dataset: str, split: str = "public",
                 limit: Optional[int] = None,
                 settings: Optional[dict] = None) -> list[Sample]:
    """加载某数据集的样本。

    Args:
        dataset: 数据集名（如 sst2）。
        split: "public" / "hidden" / "all"。
        limit: 最多返回条数。
    """
    path = samples_path(dataset, settings)
    if not path.exists():
        raise FileNotFoundError(
            f"样本文件不存在: {path}。请先运行 scripts/prepare_samples.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if split != "all":
        raw = [r for r in raw if r.get("split") == split]
    if limit is not None:
        raw = raw[:limit]
    return [Sample.from_dict(r) for r in raw]
