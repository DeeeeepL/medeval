from typing import Dict, Any, List
from data.schema import EvalDataset, Item
from clients.base import LLMClient
from judge.base import Judge
from eval.prompting import build_choice_messages, build_open_test_messages
from eval.strategies import extract_angle_answer, parse_choice_pred
from utils.text import normalize

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _parse_choice_gt_from_dataset(item: Item) -> List[str]:
    """
    根据 item.answer + item.options 解析出正确选项字母列表（支持单选/多选）。
    支持格式：
      1) answer: ["A","C","D"]
      2) answer: "A,C,D" / "A，C，D" / "A;C;D"
      3) answer: "选项1文本[SEP]选项3文本"
      4) answer: "某个选项的完整文本"（单选）
    """
    options = item.options
    ans = item.answer
    if not options or ans is None:
        return []

    import re
    # 1) list -> 多个片段
    if isinstance(ans, list):
        parts = [str(x).strip() for x in ans if str(x).strip()]
    else:
        s = str(ans).strip()
        if not s:
            return []
        if "[SEP]" in s:
            parts = [p.strip() for p in s.split("[SEP]") if p.strip()]
        elif any(sep in s for sep in [",", "，", ";", "；"]):
            parts = [p.strip() for p in re.split(r"[,，;；]", s) if p.strip()]
        else:
            parts = [s]

    correct_indices = set()
    for part in parts:
        # 单个字母
        if len(part) == 1 and part.upper() in LETTERS[:len(options)]:
            correct_indices.add(LETTERS.index(part.upper()))
            continue

        # 选项文本匹配
        matched = False
        for idx, opt in enumerate(options):
            if normalize(part) == normalize(opt):
                correct_indices.add(idx)
                matched = True
                break
        if not matched:
            # 容错：包含关系
            for idx, opt in enumerate(options):
                if normalize(part) in normalize(opt) or normalize(opt) in normalize(part):
                    correct_indices.add(idx)
                    matched = True
                    break
        if not matched:
            print(f"⚠️ 无法在选项中为题目 {item.question_id} 匹配答案片段：{part}")

    return [LETTERS[i] for i in sorted(correct_indices)]


def evaluate_choice_item(client: LLMClient,
                         judge: Judge,
                         item: Item,
                         test_model: str) -> Dict[str, Any]:
    """统一处理 single_choice / multi_choice。"""
    options = item.options
    gt_letters = _parse_choice_gt_from_dataset(item)
    full_score = item.metadata.score

    messages = build_choice_messages(item, options)
    raw = client.chat(messages, model=test_model)
    pred_letters = parse_choice_pred(raw, len(options))
    sc = judge.score_single_choice(gt_letters, pred_letters, full_score)

    return {
        "question_id": item.question_id,
        "type": item.metadata.type,   # single_choice 或 multi_choice
        "question": item.question,
        "options": options,
        "gt_letters": gt_letters,
        "pred_letters": pred_letters,
        "pred_raw": raw,
        "score_obtained": sc["score"],
        "score_full": full_score,
        "ok": sc.get("ok", False),
    }


def evaluate_open_item(client: LLMClient,
                       judge: Judge,
                       item: Item,
                       test_model: str) -> Dict[str, Any]:
    md = item.metadata
    full_score = md.score

    # 1) 待测模型回答
    messages = build_open_test_messages(item)
    raw = client.chat(messages, model=test_model)
    student_answer = extract_angle_answer(raw)

    # 2) 裁判模型按 scoring points 给 flag，本地算总分
    sc = judge.score_open_response(
        question=item.question,
        positive_points=md.positive_scoring_points,
        negative_points=md.negative_scoring_points,
        student_answer=student_answer,
        total_score=full_score,
    )

    return {
        "question_id": item.question_id,
        "type": md.type,
        "question": item.question,
        "student_answer": student_answer,
        "student_raw": raw,
        "score_obtained": sc["score"],
        "score_full": full_score,
        "ok": sc.get("ok", False),
        "scoring_points_flags": sc.get("scoring_points_flags", []),
        "judge_raw": sc.get("judge_raw", None),
    }


def run_eval(dataset: EvalDataset,
             client: LLMClient,
             judge: Judge,
             test_model: str) -> Dict[str, Any]:
    """
    对“一个数据集”评测：
    - 每道题根据 item.metadata.type 决定走选择题/问答题逻辑
    - 数据集内可以混合 single_choice / multi_choice / open_response
    """
    records: List[Dict[str, Any]] = []

    for item in dataset.dataset:
        t = item.metadata.type
        if t in ("single_choice", "multi_choice"):
            rec = evaluate_choice_item(client, judge, item, test_model)
        elif t == "open_response":
            rec = evaluate_open_item(client, judge, item, test_model)
        else:
            # 其它类型先跳过，需要时再扩展
            continue
        records.append(rec)

    total = sum(r.get("score_obtained", 0) for r in records)
    full = sum(r.get("score_full", 0) for r in records)

    def _acc(recs: List[Dict[str, Any]], typ: str) -> float:
        xs = [r for r in recs if r.get("type") == typ]
        if not xs:
            return 0.0
        ok = sum(1 for r in xs if r.get("ok"))
        return ok / len(xs)

    def _full_open(recs: List[Dict[str, Any]]) -> float:
        xs = [r for r in recs if r.get("type") == "open_response"]
        if not xs:
            return 0.0
        ok = sum(1 for r in xs if r.get("ok"))
        return ok / len(xs)

    return {
        "summary": {
            "dataset_id": dataset.dataset_metadata.dataset_id,
            "dataset_name": dataset.dataset_metadata.dataset_name,
            "num_items": len(records),
            "total_score": total,
            "max_score": full,
            "accuracy_single_choice": _acc(records, "single_choice"),
            "accuracy_multi_choice": _acc(records, "multi_choice"),
            "full_score_rate_open": _full_open(records),
        },
        "records": records,
    }
