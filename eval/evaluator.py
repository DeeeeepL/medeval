from typing import Dict, Any, List
from ..data.schema import EvalDataset, Item
from ..clients.base import LLMClient
from ..judge.base import Judge
from .prompting import build_choice_messages, build_open_test_messages
from .strategies import extract_angle_answer, parse_choice_pred
from ..utils.text import normalize

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


import re
from ..utils.text import normalize
from ..data.schema import Item

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _parse_choice_gt_from_dataset(item: Item) -> List[str]:
    """
    根据 item.answer + item.options 解析出正确选项字母列表（支持单选/多选）。
    支持格式：
      1) answer: ["A","C","D"]
      2) answer: "A,C,D" / "A，C，D" / "A;C;D"
      3) answer: "选项1文本[SEP]选项3文本"
      4) answer: "某个选项的完整文本"
    """
    options = item.options
    ans = item.answer
    if not options or ans is None:
        return []

    # 1) 若是列表，先当作“多个片段”
    if isinstance(ans, list):
        parts = [str(x).strip() for x in ans if str(x).strip()]
    else:
        s = str(ans).strip()
        if not s:
            return []
        # 2) [SEP] 拆分
        if "[SEP]" in s:
            parts = [p.strip() for p in s.split("[SEP]") if p.strip()]
        # 3) 多个字母形式 A,B,C
        elif any(sep in s for sep in [",", "，", ";", "；"]):
            parts = [p.strip() for p in re.split(r"[,，;；]", s) if p.strip()]
        else:
            parts = [s]

    correct_indices = set()

    for part in parts:
        # 如果是单个字母
        if len(part) == 1 and part.upper() in LETTERS[:len(options)]:
            correct_indices.add(LETTERS.index(part.upper()))
            continue

        # 尝试按选项文本匹配
        matched = False
        for idx, opt in enumerate(options):
            if normalize(part) == normalize(opt):
                correct_indices.add(idx)
                matched = True
                break
        if not matched:
            # 兜底：包含关系
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
    options = item.options
    gt_letters = _parse_choice_gt_from_dataset(item)
    full_score = item.metadata.score

    messages = build_choice_messages(item, options)
    raw = client.chat(messages, model=test_model)
    pred_letters = parse_choice_pred(raw, len(options))
    sc = judge.score_single_choice(gt_letters, pred_letters, full_score)

    return {
        "question_id": item.question_id,
        "type": item.metadata.type,   # 这里会是 single_choice 或 multi_choice
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
        if t in ("single_choice", "multi_choice"):
            rec = evaluate_choice_item(client, judge, item, test_model)
        elif t == "open_response":
            rec = evaluate_open_item(client, judge, item, test_model)
        else:
            # 其它类型先跳过，后面再扩展
            continue
        records.append(rec)

    total = sum(r.get("score_obtained", 0) for r in records)
    full = sum(r.get("score_full", 0) for r in records)
    acc_choice = _acc(records, "single_choice") + _acc(records, "multi_choice")
    # 上面这行如果你想分开统计，可以写两个字段，这里只是示意

    full_open = _full_open(records)

    return {
        "summary": {
            "num_items": len(records),
            "total_score": total,
            "max_score": full,
            "accuracy_single_choice": _acc(records, "single_choice"),
            "accuracy_multi_choice": _acc(records, "multi_choice"),
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
