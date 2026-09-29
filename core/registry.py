"""攻击/防御插件注册表。

实现"攻击方法与防御算法解耦"：每个插件自注册，按名加载，任意组合对阵。
"""
from __future__ import annotations

from typing import Callable, Type

ATTACKS: dict[str, Type] = {}
DEFENSES: dict[str, Type] = {}


def register_attack(name: str) -> Callable[[Type], Type]:
    """装饰器：注册一个攻击插件。"""

    def deco(cls: Type) -> Type:
        if name in ATTACKS:
            raise ValueError(f"攻击插件已存在: {name}")
        ATTACKS[name] = cls
        cls.name = name
        return cls

    return deco


def register_defense(name: str) -> Callable[[Type], Type]:
    """装饰器：注册一个防御插件。"""

    def deco(cls: Type) -> Type:
        if name in DEFENSES:
            raise ValueError(f"防御插件已存在: {name}")
        DEFENSES[name] = cls
        cls.name = name
        return cls

    return deco


def get_attack_cls(name: str) -> Type:
    if name not in ATTACKS:
        raise KeyError(
            f"未知攻击插件: {name}。可用: {list(ATTACKS.keys())}"
        )
    return ATTACKS[name]


def get_defense_cls(name: str) -> Type:
    if name not in DEFENSES:
        raise KeyError(
            f"未知防御插件: {name}。可用: {list(DEFENSES.keys())}"
        )
    return DEFENSES[name]


def list_attacks() -> list[str]:
    # 触发各 attacks 模块导入以完成注册
    import attacks  # noqa: F401
    return sorted(ATTACKS.keys())


def list_defenses() -> list[str]:
    import defenses  # noqa: F401
    return sorted(DEFENSES.keys())
