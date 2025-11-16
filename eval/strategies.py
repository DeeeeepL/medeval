import re
from typing import List

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def extract_angle_answer(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"<\s*(.*?)\s*>", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text.strip()

def parse_choice_pred(ans: str, num_options: int) -> List[str]:
    """
    把模型答案解析为 ['A'] 或 ['A','C']。
    支持:
    - <A,B> / A,B / A
    - <1,2> / 1,2 / 1
    """
    from .strategies import extract_angle_answer as _ea  # 避免循环导入
    s = _ea(ans)
    if not s:
        return []
    s = s.replace("，", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    letters = []
    for p in parts:
        if len(p) == 1 and p.upper() in LETTERS[:num_options]:
            letters.append(p.upper())
        elif p.isdigit():
            idx = int(p)
            if 1 <= idx <= num_options:
                letters.append(LETTERS[idx-1])
        else:
            ch = p[0].upper()
            if ch in LETTERS[:num_options]:
                letters.append(ch)
    return sorted(set(letters))
