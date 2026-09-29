"""运行单场攻防对阵。

用法:
    python scripts/run_match.py --dataset sst2 --attack adaptive --defense sandwich --num 10
    python scripts/run_match.py --dataset rte --attack combine --defense llm_filter --split public --mock
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli_utils import build_model, build_attack_fn, build_defense_fn
from core.arena import Arena
from core.config import load_settings
from core.data import load_samples
from core.registry import list_attacks, list_defenses


def main():
    ap = argparse.ArgumentParser(description="单场攻防对阵")
    ap.add_argument("--dataset", required=True, help="数据集名，如 sst2")
    ap.add_argument("--attack", default="adaptive", help="攻击插件名")
    ap.add_argument("--defense", default="sandwich", help="防御插件名")
    ap.add_argument("--num", type=int, default=10, help="样本数")
    ap.add_argument("--split", default="public", help="public / hidden / all")
    ap.add_argument("--mock", action="store_true", help="使用 MockModel 演练")
    ap.add_argument("--seed", type=int, default=None, help="attack_token 随机种子")
    ap.add_argument("--out", default="", help="结果保存路径（json），默认 results/match_*.json")
    args = ap.parse_args()

    # 校验插件名
    if args.attack not in list_attacks():
        sys.exit(f"未知攻击: {args.attack}，可用: {list_attacks()}")
    if args.defense not in list_defenses():
        sys.exit(f"未知防御: {args.defense}，可用: {list_defenses()}")

    settings = load_settings()
    contest = settings.get("contest", {})
    model = build_model(mock=args.mock)
    arena = Arena(
        model=model,
        max_queries=int(contest.get("max_queries", 6)),
        max_payload_bytes=int(contest.get("max_payload_bytes", 1024)),
        attack_token_format=contest.get("attack_token_format", "XXXX-XXXX-NNNN"),
        seed=args.seed,
    )

    samples = load_samples(args.dataset, split=args.split, limit=args.num)
    if not samples:
        sys.exit(f"无样本: dataset={args.dataset} split={args.split}")

    attack_fn = build_attack_fn(args.attack)
    defend_fn = build_defense_fn(args.defense)

    print(f"\n=== 对阵: {args.dataset} | attack={args.attack} | defense={args.defense} ===")
    print(f"样本数: {len(samples)} (split={args.split})\n")

    summary = arena.run_match(samples, attack_fn, defend_fn, verbose=True)

    print("\n=== 汇总 ===")
    print(f"  攻击成功率 ASR      : {summary['asr']:.4f}")
    print(f"  防御成功率 DSR      : {summary['dsr']:.4f}")
    print(f"  平局率              : {summary['draw_rate']:.4f}")
    print(f"  token 泄露率        : {summary['token_leak_rate']:.4f}")
    print(f"  任务正确率(含被劫持): {summary['task_accuracy']:.4f}")

    # 保存
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = args.out or str(
        out_dir / f"match_{args.dataset}_{args.attack}_vs_{args.defense}.json"
    )
    payload = {
        "config": {
            "dataset": args.dataset, "attack": args.attack, "defense": args.defense,
            "num": len(samples), "split": args.split, "mock": args.mock,
        },
        "summary": {k: v for k, v in summary.items() if k != "results"},
        "results": summary["results"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
