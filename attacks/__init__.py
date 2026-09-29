"""攻击插件包。

所有攻击插件通过 @register_attack 装饰器自注册。导入本包即完成注册。
可用攻击：naive, ignore, escape, fake_comp, combine, adaptive

自定义攻击：继承 BaseAttacker，实现 attack(env) -> str，加 @register_attack("your_name")。
"""
from .base import (  # noqa: F401
    BaseAttacker,
    NaiveAttacker,
    IgnoreAttacker,
    EscapeCharAttacker,
    FakeCompAttacker,
    CombineAttacker,
)
from .adaptive import AdaptiveAttacker  # noqa: F401


def get_attack(name: str) -> BaseAttacker:
    """按名实例化一个攻击插件。"""
    from core.registry import get_attack_cls
    return get_attack_cls(name)()
