"""下载核心分类数据集到本地。

将 sst2 / sms_spam / hsol / rte / mrpc 通过 HuggingFace datasets 下载并缓存到
data/raw/<dataset>.jsonl，运行时不再依赖网络。

用法:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --datasets sst2,rte --max 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许从项目根导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 控制台 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from core.config import load_settings, load_task_config, list_datasets, resolve_path


def download_one(dataset: str, max_examples: int | None = None) -> int:
    """下载单个数据集，写入 data/raw/<dataset>.jsonl。返回写入条数。"""
    from datasets import load_dataset

    cfg = load_task_config(dataset)
    settings = load_settings()
    raw_dir = resolve_path(settings["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{dataset}.jsonl"

    repo = cfg["hf_repo"]
    hf_config = cfg.get("hf_config")
    split = cfg.get("hf_split", "train")

    print(f"[{dataset}] load_dataset(repo={repo!r}, config={hf_config!r}, split={split!r}) ...")
    if hf_config:
        ds = load_dataset(repo, hf_config, split=split)
    else:
        ds = load_dataset(repo, split=split)

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            if max_examples is not None and count >= max_examples:
                break
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
    print(f"[{dataset}] 写入 {count} 条 -> {out_path}")
    return count


def main():
    ap = argparse.ArgumentParser(description="下载核心分类数据集到本地")
    ap.add_argument("--datasets", default="", help="逗号分隔的数据集名，默认全部")
    ap.add_argument("--max", type=int, default=None, help="每个数据集最多下载条数")
    args = ap.parse_args()

    if args.datasets.strip():
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = list_datasets()

    print("待下载数据集:", datasets)
    summary = {}
    for d in datasets:
        try:
            summary[d] = download_one(d, args.max)
        except Exception as e:
            print(f"[{d}] 下载失败: {type(e).__name__}: {e}")
            summary[d] = -1

    print("\n=== 下载汇总 ===")
    for d, n in summary.items():
        print(f"  {d}: {n if n >= 0 else 'FAILED'}")


if __name__ == "__main__":
    main()
