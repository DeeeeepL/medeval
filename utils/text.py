import re

def normalize(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip().lower()
