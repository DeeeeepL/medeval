# medeval/judge/llm_judge.py
import json
from typing import Dict, Any, List
from .base import Judge
from data.schema import ScoringPoint
from clients.base import LLMClient

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
    - 只持有“裁判模型 client”（可以是 gpt-4o，也可以是别的）
    - client 内部已经配置了默认模型，无需在这里传 model 名
    """

    def __init__(self, judge_client: LLMClient):
        self.judge_client = judge_client

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

        raw = self.judge_client.chat(messages)  # 不再传 model，使用裁判 client 默认模型
        flags = self._parse_flags(raw, positive_points, negative_points)
        scoring_points_flags = flags["scoring_points_flags"]

        # 本地算分
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
        try:
            j = self._safe_json_loads(raw)
        except Exception:
            # 全部 false 兜底
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
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
        return json.loads(text)
