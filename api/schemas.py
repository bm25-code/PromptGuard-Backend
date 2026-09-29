"""FastAPI 请求/响应模型。"""
from __future__ import annotations

from typing import Optional, Any
from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    dataset: str = Field(..., example="sst2")
    attack: str = Field("adaptive", example="adaptive")
    defense: str = Field("sandwich", example="sandwich")
    num: int = Field(10, ge=1, le=500)
    split: str = Field("public")
    mock: Optional[bool] = None
    seed: Optional[int] = None


class TournamentRequest(BaseModel):
    datasets: Optional[list[str]] = None
    attacks: Optional[list[str]] = None
    defenses: Optional[list[str]] = None
    num: int = Field(10, ge=1, le=500)
    split: str = Field("public")
    mock: Optional[bool] = None
    seed: Optional[int] = None


class AttackPreviewRequest(BaseModel):
    """对指定样本运行攻击，返回 payload 与探测历史。"""
    target_instruction: str
    input: str
    attack_token: str
    attack: str = "adaptive"
    defense: str = "none"
    mock: Optional[bool] = None
    seed: Optional[int] = None


class DefendPreviewRequest(BaseModel):
    """对给定 final_prompt 应用防御，返回处理后提示词。"""
    final_prompt: str
    defense: str = "sandwich"
    target_task: str = ""


class SampleOut(BaseModel):
    target_instruction: str
    input: str
    output: str
    label: Optional[int] = None
    dataset: str = ""
    split: str = "public"
    index: int = 0
