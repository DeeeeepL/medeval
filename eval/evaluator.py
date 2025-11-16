from typing import Dict, Any, List, Optional
from data.schema import EvalDataset, Item
from clients.base import LLMClient
from judge.base import Judge
from eval.prompting import build_choice_messages, build_open_test_messages
from eval.strategies import extract_angle_answer, parse_choice_pred
from utils.text import normalize
from eval.choice_aug import (
    make_base_variant,
    make_shuffle_variant,
    make_nota_variant,
)
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
                         test_model: str,
                         variant: str = "base",
                         shuffle_seed: int = 0) -> Dict[str, Any]:
    """
    统一处理 single_choice / multi_choice，不同 variant：
      - base   : 原题
      - shuffle: 打乱选项
      - nota   : NOTA 题（以上皆非）
    """
    base_options = item.options
    base_gt_letters = _parse_choice_gt_from_dataset(item)
    full_score = item.metadata.score

    extra = {}

    if variant == "base":
        options, gt_letters = make_base_variant(base_options, base_gt_letters)
    elif variant == "shuffle":
        options, gt_letters, extra = make_shuffle_variant(
            base_options, base_gt_letters, seed=shuffle_seed
        )
    elif variant == "nota":
        options, gt_letters, extra = make_nota_variant(base_options, base_gt_letters)
    else:
        # 未知模式，退回 base
        options, gt_letters = make_base_variant(base_options, base_gt_letters)

    # 构造选择题 prompt（用增强后的 options）
    messages = build_choice_messages(item, options)
    raw = client.chat(messages, model=test_model)
    pred_letters = parse_choice_pred(raw, len(options))

    sc = judge.score_single_choice(gt_letters, pred_letters, full_score)

    rec: Dict[str, Any] = {
        "question_id": item.question_id,
        "type": item.metadata.type,   # single_choice / multi_choice
        "variant": variant,           # base / shuffle / nota
        "question": item.question,
        "options": options,
        "gt_letters": gt_letters,
        "pred_letters": pred_letters,
        "pred_raw": raw,
        "score_obtained": sc["score"],
        "score_full": full_score,
        "ok": sc.get("ok", False),
    }
    if extra:
        rec["augment_extra"] = extra
    return rec

def evaluate_open_item(client: LLMClient,
                       judge: Judge,
                       item: Item,
                       test_model: str) -> Dict[str, Any]:
    md = item.metadata
    full_score = md.score

    # 1) 待测模型回答
    messages = build_open_test_messages(item)
    raw = client.chat(messages, model=test_model)
    answer = extract_angle_answer(raw)

    # 2) 裁判模型按 scoring points 给 flag，本地算总分
    sc = judge.score_open_response(
        question=item.question,
        positive_points=md.positive_scoring_points,
        negative_points=md.negative_scoring_points,
        answer=answer,
        total_score=full_score,
    )

    return {
        "question_id": item.question_id,
        "type": md.type,
        "question": item.question,
        "answer": answer,
        "raw": raw,
        "score_obtained": sc["score"],
        "score_full": full_score,
        "ok": sc.get("ok", False),
        "scoring_points_flags": sc.get("scoring_points_flags", []),
        "judge_raw": sc.get("judge_raw", None),
    }


def run_eval(dataset: EvalDataset,
             client: LLMClient,
             judge: Judge,
             test_model: str,
             choice_modes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    对一个数据集评测：
      - choice_modes 指定选择题评测模式：
        ["base"]                -> 只测原题
        ["base", "shuffle"]     -> 原题 + 打乱
        ["base", "nota"]        -> 原题 + NOTA
        ["base", "shuffle", "nota"] -> 三种都测
    """
    if choice_modes is None:
        choice_modes = ["base"]

    # 去重 & 处理 "all"
    if "all" in choice_modes:
        choice_modes = ["base", "shuffle", "nota"]
    choice_modes = list(dict.fromkeys(choice_modes))  # 保持顺序去重

    records: List[Dict[str, Any]] = []

    for item in dataset.dataset:
        t = item.metadata.type

        if t in ("single_choice", "multi_choice", "multiple_choice"):
            # 对每个 variant 都跑一遍
            for mode in choice_modes:
                rec = evaluate_choice_item(
                    client, judge, item, test_model, variant=mode
                )
                records.append(rec)

        elif t == "open_response":
            rec = evaluate_open_item(client, judge, item, test_model)
            records.append(rec)

        else:
            # 其它题型先跳过
            continue

    # -------- 下面 summary 你可以保持简单，先汇总总体 --------
    total = sum(r.get("score_obtained", 0) for r in records)
    full = sum(r.get("score_full", 0) for r in records)

    def _acc(recs: List[Dict[str, Any]], typ: str, variant: Optional[str] = None) -> float:
        xs = [r for r in recs if r.get("type") == typ]
        if variant is not None:
            xs = [r for r in xs if r.get("variant") == variant]
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

    # 也可以把各个 variant 的选择题准确率单独放出来
    choice_summary = {}
    for mode in choice_modes:
        choice_summary[mode] = {
            "accuracy_single_choice": _acc(records, "single_choice", variant=mode),
            "accuracy_multi_choice": _acc(records, "multi_choice", variant=mode)
                                 + _acc(records, "multiple_choice", variant=mode),
        }

    return {
        "summary": {
            "dataset_id": dataset.dataset_metadata.dataset_id,
            "dataset_name": dataset.dataset_metadata.dataset_name,
            "num_records": len(records),
            "total_score": total,
            "max_score": full,
            "choice_summary": choice_summary,
            "full_score_rate_open": _full_open(records),
        },
        "records": records,
    }
