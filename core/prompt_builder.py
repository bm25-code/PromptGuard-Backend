"""基线 PromptBuilder。

赛题固定的基线处理流程：

    target_instruction
            │
            ▼
        PromptBuilder
            ▲
            │
        data_prompt   (= clean_data + [INJECT])
            │
            ▼
           LLM

未部署防御时（赛题给定）：

    def query(target_instruction, data_prompt):
        prompt = target_instruction + "\\n\\n" + "Input:\\n" + data_prompt
        return model.query(prompt)
"""
from __future__ import annotations


class PromptBuilder:
    """构造最终提示词：target_instruction + "\\n\\nInput:\\n" + data_prompt。

    data_prompt 由 clean_data 与攻击方注入 payload 拼接而成。
    """

    SEPARATOR = "\n\nInput:\n"

    def build(self, target_instruction: str, data_prompt: str) -> str:
        return f"{target_instruction}{self.SEPARATOR}{data_prompt}"

    @staticmethod
    def split(final_prompt: str) -> tuple[str, str]:
        """将最终提示词拆回 (target_instruction, data_prompt)。

        防御方只能拿到 final_prompt；若需要重组结构，可据此拆分。
        找不到分隔符时返回 (final_prompt, "")。
        """
        sep = PromptBuilder.SEPARATOR
        idx = final_prompt.find(sep)
        if idx == -1:
            return final_prompt, ""
        instruction = final_prompt[:idx]
        data = final_prompt[idx + len(sep):]
        return instruction, data
