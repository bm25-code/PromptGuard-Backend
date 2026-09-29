"""赛题样本数据模型：{target_instruction, input, output}。"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Sample:
    """一条赛题测试样本。

    target_instruction: 可信任务指令（系统提示词）。
    input: 合法原始数据（clean_data），不含任何注入。
    output: 任务正确答案文本（ground truth）。
    label: 数值标签（用于评估），可选。
    dataset: 所属数据集名。
    split: public / hidden。
    """
    target_instruction: str
    input: str
    output: str
    label: int | None = None
    dataset: str = ""
    split: str = "public"
    index: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(
            target_instruction=d["target_instruction"],
            input=d["input"],
            output=d["output"],
            label=d.get("label"),
            dataset=d.get("dataset", ""),
            split=d.get("split", "public"),
            index=d.get("index", 0),
        )
