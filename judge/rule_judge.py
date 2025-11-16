from typing import Dict, Any, List
from .base import Judge
from ..data.schema import ScoringPoint

class RuleJudge(Judge):
    """
    规则裁判：
    - single_choice / multi_choice: 全对得分，否则 0
    - open_response: 简单子串匹配 positive / negative，计算得分
    """

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
        ans = (student_answer or "").lower()
        scoring_points_flags = []

        score = 0
        # positive
        for p in positive_points:
            hit = p.criterion.lower() in ans
            if hit:
                score += p.points
            scoring_points_flags.append({
                "criterion": p.criterion,
                "points": p.points,
                "flag": bool(hit),
            })

        # negative
        for n in negative_points:
            hit = n.criterion.lower() in ans
            if hit:
                score += n.points  # 注意: points 是负数
            scoring_points_flags.append({
                "criterion": n.criterion,
                "points": n.points,
                "flag": bool(hit),
            })

        # 裁剪
        score = max(0, min(score, total_score))

        return {
            "score": score,
            "ok": score == total_score,
            "scoring_points_flags": scoring_points_flags,
        }
