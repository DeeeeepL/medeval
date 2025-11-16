from typing import Dict, Any, List
from ..data.schema import EvalDataset, Item
from ..clients.base import LLMClient
from ..judge.base import Judge
from .prompting import build_choice_messages, build_open_test_messages
from .strategies import extract_angle_answer, parse_choice_pred
from ..utils.text import normalize

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _parse_choice_gt_from_dataset(item: Item) -> List[str]:
    """
    针对你的选择题数据：
    - 如果 answer 是选项文本，就定位对应字母
    - 如果 answer 是单个字母，直接返回
    - 如果未来扩展多选 + [SEP]，可在此做拆分
    """
    options = item.options
    ans = str(item.answer).strip()
    if not options:
        return []

    # 单字母
    if len(ans) == 1 and ans.upper() in LETTERS[:len(options)]:
        return [ans.upper()]

    # 尝试与选项文本匹配
    for i, opt in enumerate(options):
        if normalize(ans) == normalize(opt):
            return [LETTERS[i]]

    # 简单兜底：没有匹配就返回空
    return []


def evaluate_single_choice_item(client: LLMClient,
                                judge: Judge,
                                item: Item,
                                test_model: str) -> Dict[str, Any]:
    options = item.options
    gt_letters = _parse_choice_gt_from_dataset(item)
    full_score = item.metadata.score

    messages = build_choice_messages(item, options)
    raw = client.chat(messages, model=test_model)
    pred_letters = parse_choice_pred(raw, len(options))
    sc = judge.score_single_choice(gt_letters, pred_letters, full_score)

    return {
        "question_id": item.question_id,
        "type": item.metadata.type,
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

    # 2) 裁判模型给 flag + 本地算分
    sc = judge.score_open_response(
        question=item.question,
        positive_points=md.positive_scoring_points,
        negative_points=md.negative_scoring_points,
        student_answer=student_answer,
        total_score=full_score
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
    records: List[Dict[str, Any]] = []

    for item in dataset.dataset:
        t = item.metadata.type
        if t == "single_choice":
            rec = evaluate_single_choice_item(client, judge, item, test_model)
        elif t == "open_response":
            rec = evaluate_open_item(client, judge, item, test_model)
        else:
            continue
        records.append(rec)

    total = sum(r.get("score_obtained", 0) for r in records)
    full = sum(r.get("score_full", 0) for r in records)
    acc_choice = _acc(records, "single_choice")
    full_open = _full_open(records)

    return {
        "summary": {
            "num_items": len(records),
            "total_score": total,
            "max_score": full,
            "accuracy_single_choice": acc_choice,
            "full_score_rate_open": full_open,
        },
        "records": records,
    }


def _acc(records: List[Dict[str, Any]], typ: str) -> float:
    xs = [r for r in records if r.get("type") == typ]
    if not xs:
        return 0.0
    ok = sum(1 for r in xs if r.get("ok"))
    return ok / len(xs)


def _full_open(records: List[Dict[str, Any]]) -> float:
    xs = [r for r in records if r.get("type") == "open_response"]
    if not xs:
        return 0.0
    ok = sum(1 for r in xs if r.get("ok"))
    return ok / len(xs)
