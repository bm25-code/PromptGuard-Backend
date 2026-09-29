"""FastAPI 服务层。

端点:
  GET  /health               健康检查
  GET  /attacks              列出攻击插件
  GET  /defenses             列出防御插件
  GET  /datasets             列出可用数据集
  GET  /samples/{dataset}    查看样本
  POST /match                单场对阵
  POST /tournament           循环赛
  POST /preview/attack       攻击预览（返回 payload + 探测历史）
  POST /preview/defend       防御预览（返回处理后提示词）
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# 项目根入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 触发插件注册
import attacks  # noqa: F401
import defenses  # noqa: F401

from fastapi import FastAPI, HTTPException, Query

from core.arena import Arena
from core.config import load_settings, model_settings
from core.data import load_samples
from core.env import DefendEnv
from core.registry import (
    list_attacks, list_defenses, get_attack_cls, get_defense_cls,
)
from core.config import list_datasets
from models import OpenAIModel, MockModel

from .schemas import (
    MatchRequest, TournamentRequest, AttackPreviewRequest,
    DefendPreviewRequest, SampleOut,
)

app = FastAPI(title="PromptGuard Backend", version="1.0.0")


def _build_model(mock: Optional[bool]):
    mcfg = model_settings()
    if mock is None:
        mock = not bool(mcfg.get("api_key"))
    if mock:
        return MockModel()
    return OpenAIModel.from_settings(mcfg)


def _build_arena(seed: Optional[int] = None, model=None) -> Arena:
    settings = load_settings()
    contest = settings.get("contest", {})
    return Arena(
        model=model or _build_model(None),
        max_queries=int(contest.get("max_queries", 6)),
        max_payload_bytes=int(contest.get("max_payload_bytes", 1024)),
        attack_token_format=contest.get("attack_token_format", "XXXX-XXXX-NNNN"),
        seed=seed,
    )


# ---- 基础端点 --------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_configured": bool(model_settings().get("api_key"))}


@app.get("/attacks")
def get_attacks():
    return {"attacks": list_attacks()}


@app.get("/defenses")
def get_defenses():
    return {"defenses": list_defenses()}


@app.get("/datasets")
def get_datasets():
    return {"datasets": list_datasets()}


@app.get("/samples/{dataset}")
def get_samples(dataset: str, split: str = "public", limit: int = 10):
    try:
        samples = load_samples(dataset, split=split, limit=limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"dataset": dataset, "split": split, "n": len(samples),
            "samples": [SampleOut(**s.__dict__).model_dump() for s in samples]}


# ---- 对阵端点 --------------------------------------------------------

@app.post("/match")
def run_match(req: MatchRequest):
    if req.attack not in list_attacks():
        raise HTTPException(400, f"未知攻击: {req.attack}")
    if req.defense not in list_defenses():
        raise HTTPException(400, f"未知防御: {req.defense}")
    try:
        samples = load_samples(req.dataset, split=req.split, limit=req.num)
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))
    if not samples:
        raise HTTPException(404, detail=f"无样本: {req.dataset}/{req.split}")

    model = _build_model(req.mock)
    arena = _build_arena(req.seed, model)
    attack_fn = get_attack_cls(req.attack)()
    defend_fn = get_defense_cls(req.defense)()
    summary = arena.run_match(samples, attack_fn, defend_fn, verbose=False)
    return {
        "config": req.model_dump(),
        "summary": {k: v for k, v in summary.items() if k != "results"},
        "results": summary["results"],
    }


@app.post("/tournament")
def run_tournament(req: TournamentRequest):
    datasets = req.datasets or list_datasets()
    attacks_ = req.attacks or list_attacks()
    defenses_ = req.defenses or list_defenses()

    model = _build_model(req.mock)
    arena = _build_arena(req.seed, model)
    matrix, detail = {}, []
    for ds in datasets:
        try:
            samples = load_samples(ds, split=req.split, limit=req.num)
        except FileNotFoundError:
            continue
        if not samples:
            continue
        matrix[ds] = {}
        for atk in attacks_:
            matrix[ds].setdefault(atk, {})
            af = get_attack_cls(atk)()
            for dfn in defenses_:
                df = get_defense_cls(dfn)()
                s = arena.run_match(samples, af, df, verbose=False)
                matrix[ds][atk][dfn] = {
                    "asr": round(s["asr"], 4), "dsr": round(s["dsr"], 4),
                    "n": s["n"],
                }
                detail.append({"dataset": ds, "attack": atk, "defense": dfn,
                               **matrix[ds][atk][dfn]})
    return {"matrix": matrix, "detail": detail}


# ---- 预览端点 --------------------------------------------------------

@app.post("/preview/attack")
def preview_attack(req: AttackPreviewRequest):
    """对单个样本运行攻击，返回 payload 与探测历史。"""
    if req.attack not in list_attacks():
        raise HTTPException(400, f"未知攻击: {req.attack}")
    model = _build_model(req.mock)
    arena = _build_arena(req.seed, model)
    defend_fn = get_defense_cls(req.defense)()
    env = Arena.__new__(Arena)  # 仅复用 gen_attack_token
    # 直接构造 AttackEnv
    from core.env import AttackEnv
    from core.prompt_builder import PromptBuilder
    settings = load_settings()
    contest = settings.get("contest", {})
    token = req.attack_token or arena.gen_attack_token()
    env = AttackEnv(
        target_instruction=req.target_instruction,
        input=req.input,
        attack_token=token,
        defend_fn=defend_fn,
        model=model,
        prompt_builder=PromptBuilder(),
        max_queries=int(contest.get("max_queries", 6)),
        max_payload_bytes=int(contest.get("max_payload_bytes", 1024)),
    )
    attacker = get_attack_cls(req.attack)()
    payload = attacker.attack(env)
    env.set_injection(payload)
    official = env.query_official()
    return {
        "attack_token": token,
        "payload": payload,
        "payload_bytes": len(payload.encode("utf-8")),
        "queries_used": env.query_count,
        "probe_history": env.history,
        "official_response": official,
    }


@app.post("/preview/defend")
def preview_defend(req: DefendPreviewRequest):
    """对给定 final_prompt 应用防御，返回处理后提示词。"""
    if req.defense not in list_defenses():
        raise HTTPException(400, f"未知防御: {req.defense}")
    defender = get_defense_cls(req.defense)()
    model = _build_model(None)
    env = DefendEnv(final_prompt=req.final_prompt, model=model,
                    target_task=req.target_task)
    processed = defender.defend(env)
    return {"processed_prompt": processed}
