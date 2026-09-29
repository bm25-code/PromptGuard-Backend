"""防御插件包。

所有防御插件通过 @register_defense 装饰器自注册。导入本包即完成注册。
可用防御：none, instructional, sandwich, delimiters, xml, random_seq,
          paraphrasing, llm_filter

自定义防御：继承 BaseDefender，实现 defend(env) -> str，加 @register_defense("name")。
"""
from .base import (  # noqa: F401
    BaseDefender,
    NoneDefense,
    InstructionalDefense,
    SandwichDefense,
    DelimitersDefense,
    XMLDefense,
    RandomSeqDefense,
)
from .llm_defenses import (  # noqa: F401
    ParaphrasingDefense,
    LLMFilterDefense,
)


def get_defense(name: str) -> BaseDefender:
    """按名实例化一个防御插件。"""
    from core.registry import get_defense_cls
    return get_defense_cls(name)()
