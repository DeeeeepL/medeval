from typing import Dict, Any, List
from ..data.schema import Item

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def build_choice_messages(item: Item, options: list[str]) -> List[Dict[str, str]]:
    md = item.metadata
    tpl = md.prompt_template or ""
    q = item.question
    letters = LETTERS[:len(options)]
    opts_str = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))

    if tpl.strip():
        content = tpl + f"\n题目：{q}\n\n{opts_str}\n"
    else:
        content = (
            "##任务：请回答以下选择题。\n"
            "要求：只输出正确选项的序号，放在尖括号 <> 中；多选题用逗号分隔，例如 <A,B>。\n\n"
            f"题目：{q}\n\n{opts_str}\n\n答：<>"
        )
    return [{"role": "user", "content": content}]


def build_open_test_messages(item: Item) -> List[Dict[str, str]]:
    md = item.metadata
    tpl = md.prompt_template or ""
    q = item.question

    if tpl.strip():
        content = tpl + f"\n题目：{q}\n答：<>"
    else:
        content = (
            "##任务：请回答下面的开放问答题，并将最终答案填写在尖括号 <> 中输出。\n"
            "不要输出多余说明。\n\n"
            f"题目：{q}\n\n答：<>"
        )
    return [{"role": "user", "content": content}]
