"""将下载的原始数据集加工为赛题格式样本。

每条样本：{target_instruction, input, output, label, dataset, split, index}
按 public:hidden = 1:1 划分（隐藏集对参赛者不可见）。

用法:
    python scripts/prepare_samples.py
    python scripts/prepare_samples.py --datasets sst2 --num 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 控制台 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from core.config import load_settings, load_task_config, list_datasets, resolve_path


def read_system_prompt(name: str, prompts_dir: Path) -> str:
    path = prompts_dir / f"{name}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_input(example: dict, cfg: dict) -> str:
    """按 input_template 渲染输入文本。"""
    template = cfg["input_template"]
    fields = cfg["input_fields"]
    try:
        return template.format(**{k: str(example.get(k, "")) for k in fields})
    except Exception:
        # 兜底：拼接所有字段
        return " ".join(str(example.get(k, "")) for k in fields)


def resolve_label(example: dict, cfg: dict) -> int:
    """解析数值标签（应用 raw_class_map，如 hsol 的 3 类 -> 2 类）。"""
    label_field = cfg.get("label_field", "class" if "raw_class_map" in cfg else "label")
    raw = example.get(label_field)
    if raw is None:
        raise ValueError(f"样本缺少标签字段 {label_field}: {example}")
    raw = int(raw)
    if "raw_class_map" in cfg:
        raw = int(cfg["raw_class_map"][raw])
    return raw


def prepare_one(dataset: str, num: int | None, prompts_dir: Path,
                samples_dir: Path, split_seed: int, public_ratio: float) -> int:
    cfg = load_task_config(dataset)
    settings = load_settings()
    raw_path = resolve_path(settings["paths"]["raw_dir"]) / f"{dataset}.jsonl"
    if not raw_path.exists():
        print(f"[{dataset}] 原始数据不存在: {raw_path}，跳过（请先运行 download_datasets.py）")
        return 0

    instruction = read_system_prompt(cfg["system_prompt"], prompts_dir)
    label_map = {int(k): v for k, v in cfg["label_map"].items()}

    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            try:
                label = resolve_label(ex, cfg)
            except Exception:
                continue
            if label not in label_map:
                continue
            inp = build_input(ex, cfg).replace("\n", " ").strip()
            if not inp:
                continue
            records.append({
                "target_instruction": instruction,
                "input": inp,
                "output": label_map[label],
                "label": label,
                "dataset": dataset,
            })
            if num is not None and len(records) >= num * 2:  # 预留 public+hidden
                break

    if not records:
        print(f"[{dataset}] 无可用样本")
        return 0

    # 均衡截断到偶数（public:hidden = 1:1）
    if num is not None:
        keep = min(len(records), num * 2)
        records = records[:keep]
    if len(records) % 2 != 0:
        records = records[:-1]

    rng = random.Random(split_seed)
    rng.shuffle(records)
    half = int(len(records) * public_ratio)
    for i, r in enumerate(records):
        r["split"] = "public" if i < half else "hidden"
        r["index"] = i if r["split"] == "public" else i - half

    out_path = samples_dir / f"{dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    pub = sum(1 for r in records if r["split"] == "public")
    hid = sum(1 for r in records if r["split"] == "hidden")
    print(f"[{dataset}] 写入 {len(records)} 条 (public={pub}, hidden={hid}) -> {out_path}")
    return len(records)


def main():
    ap = argparse.ArgumentParser(description="加工赛题格式样本")
    ap.add_argument("--datasets", default="", help="逗号分隔的数据集名，默认全部")
    ap.add_argument("--num", type=int, default=100, help="每个数据集每份(public/hidden)样本数")
    args = ap.parse_args()

    settings = load_settings()
    samples_dir = resolve_path(settings["paths"]["samples_dir"])
    samples_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = resolve_path(settings["paths"]["system_prompts_dir"])

    if args.datasets.strip():
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = list_datasets()

    split_cfg = settings.get("split", {})
    public_ratio = float(split_cfg.get("public_ratio", 0.5))
    seed = int(split_cfg.get("seed", 42))

    total = 0
    for d in datasets:
        total += prepare_one(d, args.num, prompts_dir, samples_dir, seed, public_ratio)
    print(f"\n共生成 {total} 条样本，目录: {samples_dir}")


if __name__ == "__main__":
    main()
