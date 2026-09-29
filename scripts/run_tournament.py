"""循环赛：对所有 攻击×防御×数据集 组合跑对阵，输出 ASR/DSR 矩阵。

用法:
    python scripts/run_tournament.py --datasets sst2,rte --num 20
    python scripts/run_tournament.py --attacks adaptive,combine --defenses sandwich,llm_filter --mock
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
from core.config import list_datasets as all_datasets
from core.data import load_samples
from core.registry import list_attacks, list_defenses


def main():
    ap = argparse.ArgumentParser(description="攻防循环赛")
    ap.add_argument("--datasets", default="", help="逗号分隔，默认全部")
    ap.add_argument("--attacks", default="", help="逗号分隔，默认全部")
    ap.add_argument("--defenses", default="", help="逗号分隔，默认全部")
    ap.add_argument("--num", type=int, default=10, help="每场样本数")
    ap.add_argument("--split", default="public")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="results/tournament.json")
    args = ap.parse_args()

    datasets = ([d.strip() for d in args.datasets.split(",") if d.strip()]
                or all_datasets())
    attacks = ([a.strip() for a in args.attacks.split(",") if a.strip()]
               or list_attacks())
    defenses = ([d.strip() for d in args.defenses.split(",") if d.strip()]
                or list_defenses())

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

    print(f"\n=== 循环赛: {len(datasets)} 数据集 × {len(attacks)} 攻击 × {len(defenses)} 防御 ===")
    print(f"datasets={datasets}\nattacks={attacks}\ndefenses={defenses}\n")

    # 矩阵: per dataset -> per attack -> per defense -> summary
    matrix: dict = {}
    detail: list = []
    total_matches = len(datasets) * len(attacks) * len(defenses)
    done = 0

    for ds in datasets:
        matrix[ds] = {}
        try:
            samples = load_samples(ds, split=args.split, limit=args.num)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue
        if not samples:
            print(f"[skip] {ds} 无样本")
            continue

        for atk in attacks:
            matrix[ds].setdefault(atk, {})
            attack_fn = build_attack_fn(atk)
            for dfn in defenses:
                defend_fn = build_defense_fn(dfn)
                done += 1
                print(f"[{done}/{total_matches}] {ds} | {atk} vs {dfn}")
                summary = arena.run_match(samples, attack_fn, defend_fn, verbose=False)
                matrix[ds][atk][dfn] = {
                    "asr": round(summary["asr"], 4),
                    "dsr": round(summary["dsr"], 4),
                    "token_leak_rate": round(summary["token_leak_rate"], 4),
                    "task_accuracy": round(summary["task_accuracy"], 4),
                    "n": summary["n"],
                }
                detail.append({
                    "dataset": ds, "attack": atk, "defense": dfn,
                    **matrix[ds][atk][dfn],
                })
                print(f"    ASR={summary['asr']:.3f} DSR={summary['dsr']:.3f} "
                      f"leak={summary['token_leak_rate']:.3f}")

    # 打印 ASR / DSR 矩阵表
    print("\n=== ASR 矩阵 (行=攻击, 列=防御, 汇总各数据集平均) ===")
    print_asr_matrix(attacks, defenses, matrix)
    print("\n=== DSR 矩阵 (行=防御, 列=攻击, 汇总各数据集平均) ===")
    print_dsr_matrix(defenses, attacks, matrix)

    # 平均排名（按赛题：多种攻击/防御取平均成功率）
    avg_asr_by_attack = {a: avg([matrix[ds][a][dfn]["asr"]
                                 for ds in matrix for dfn in defenses if dfn in matrix[ds].get(a, {})])
                         for a in attacks}
    avg_dsr_by_defense = {d: avg([matrix[ds][a][d]["dsr"]
                                  for ds in matrix for a in attacks if d in matrix[ds].get(a, {})])
                          for d in defenses}
    print("\n=== 攻击方平均 ASR（跨防御×数据集） ===")
    for a, v in sorted(avg_asr_by_attack.items(), key=lambda x: -x[1]):
        print(f"  {a:12s} {v:.4f}")
    print("\n=== 防御方平均 DSR（跨攻击×数据集） ===")
    for d, v in sorted(avg_dsr_by_defense.items(), key=lambda x: -x[1]):
        print(f"  {d:12s} {v:.4f}")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"matrix": matrix, "detail": detail,
                   "avg_asr_by_attack": avg_asr_by_attack,
                   "avg_dsr_by_defense": avg_dsr_by_defense}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {args.out}")


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def print_asr_matrix(attacks, defenses, matrix):
    # 行: 攻击, 列: 防御
    label = "attack\\defense"
    header = f"{label:>16s}" + "".join(f"{d:>16s}" for d in defenses)
    print(header)
    for a in attacks:
        row = f"{a:>16s}"
        for d in defenses:
            vals = [matrix[ds][a][d]["asr"] for ds in matrix if a in matrix[ds] and d in matrix[ds][a]]
            v = avg(vals)
            row += f"{v:>16.3f}"
        print(row)


def print_dsr_matrix(defenses, attacks, matrix):
    label = "defense\\attack"
    header = f"{label:>16s}" + "".join(f"{a:>16s}" for a in attacks)
    print(header)
    for d in defenses:
        row = f"{d:>16s}"
        for a in attacks:
            vals = [matrix[ds][a][d]["dsr"] for ds in matrix if a in matrix[ds] and d in matrix[ds][a]]
            v = avg(vals)
            row += f"{v:>16.3f}"
        print(row)


if __name__ == "__main__":
    main()
