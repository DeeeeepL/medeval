import json
from typing import Dict, Any, List
from .base import Judge
from ..data.schema import ScoringPoint
from ..clients.base import LLMClient

JUDGE_SYSTEM_PROMPT = """
You are a strict medical grading assistant for thoracic surgery exam questions.

You will receive:
1) The exam question
2) The student's answer
3) A grading rubric with:
   - positive scoring points (criterion + positive points)
   - negative scoring points (criterion + negative points)

Your task:
- For EACH scoring point (both positive and negative), decide whether the student's answer satisfies that criterion (flag = true or false).
- Output ONLY a JSON object with this exact structure:

{
  "positive": [
    {"criterion": "...", "flag": true/false},
    ...
  ],
  "negative": [
    {"criterion": "...", "flag": true/false},
    ...
  ]
}

Rules:
- Do NOT compute the final score. Only decide flags.
- criterion strings in the output MUST exactly copy those from the rubric.
- No extra keys. No comments. No explanation outside this JSON.
""".strip()


class LLMJudge(Judge):
    """
    GPT-4o 裁判：
    - 返回每个 scoring point 的 flag
    - 得分完全由本地 Python 按 points 计算（正负皆可）
    """

    def __init__(self, client: LLMClient, judge_model: str = "gpt-4o"):
        self.client = client
        self.judge_model = judge_model

    def score_single_choice(self,
                            gt_letters: List[str],
                            pred_letters: List[str],
                            total_score: int) -> Dict[str, Any]:
        gt_set = set(gt_letters)
        pred_set = set(pred_letters)
        ok = gt_set == pred_set and len(gt_set) > 0
        return {
            "score": total_score if ok else 0,
            "ok": ok,
        }

    def score_open_response(self,
                            question: str,
                            positive_points: List[ScoringPoint],
                            negative_points: List[ScoringPoint],
                            student_answer: str,
                            total_score: int) -> Dict[str, Any]:
        rubric_lines = ["Positive scoring points:"]
        for p in positive_points:
            rubric_lines.append(f"- (+{p.points}) {p.criterion}")
        rubric_lines.append("\nNegative scoring points:")
        for n in negative_points:
            rubric_lines.append(f"- ({n.points}) {n.criterion}")
        rubric_text = "\n".join(rubric_lines)

        user_content = f"""Question:
{question}

Student Answer:
{student_answer}

Grading Rubric:
{rubric_text}

Remember: ONLY output JSON with positive[] and negative[].
"""

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = self.client.chat(messages, model=self.judge_model)
        flags = self._parse_flags(raw, positive_points, negative_points)
        scoring_points_flags = flags["scoring_points_flags"]

        # 本地计算得分
        score = 0
        for sp in scoring_points_flags:
            if sp["flag"]:
                score += sp["points"]
        score = max(0, min(score, total_score))

        return {
            "score": score,
            "ok": score == total_score,
            "scoring_points_flags": scoring_points_flags,
            "judge_raw": raw,
        }

    def _parse_flags(self,
                     raw: str,
                     positive_points: List[ScoringPoint],
                     negative_points: List[ScoringPoint]) -> Dict[str, Any]:
        """
        解析 GPT-4o 输出的 JSON，并与 rubric 对齐：
        - 若缺失某个 criterion，就默认 flag=False
        """
        try:
            j = self._safe_json_loads(raw)
        except Exception:
            # 解析失败，全部 false
            scoring_points_flags = []
            for p in positive_points:
                scoring_points_flags.append({
                    "criterion": p.criterion,
                    "points": p.points,
                    "flag": False,
                })
            for n in negative_points:
                scoring_points_flags.append({
                    "criterion": n.criterion,
                    "points": n.points,
                    "flag": False,
                })
            return {"scoring_points_flags": scoring_points_flags}

        pos_flags = {d.get("criterion", ""): bool(d.get("flag", False))
                     for d in j.get("positive", []) or []}
        neg_flags = {d.get("criterion", ""): bool(d.get("flag", False))
                     for d in j.get("negative", []) or []}

        scoring_points_flags = []
        for p in positive_points:
            scoring_points_flags.append({
                "criterion": p.criterion,
                "points": p.points,
                "flag": bool(pos_flags.get(p.criterion, False)),
            })
        for n in negative_points:
            scoring_points_flags.append({
                "criterion": n.criterion,
                "points": n.points,
                "flag": bool(neg_flags.get(n.criterion, False)),
            })

        return {"scoring_points_flags": scoring_points_flags}

    @staticmethod
    def _safe_json_loads(text: str) -> Any:
        # 提取第一个看起来像 JSON 的对象
        text = text.strip()
        # 简单处理：找到第一个 '{' 和最后一个 '}' 之间
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
        return json.loads(text)
