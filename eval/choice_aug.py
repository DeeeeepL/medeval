# medeval/eval/choice_aug.py
# -*- coding: utf-8 -*-
"""
选择题数据增强：base / shuffle / nota

- base   : 不改动选项
- shuffle: 打乱选项及对应答案
- nota   : 将原正确选项移除，新增“以上皆非/None of the above”为正确答案
"""

from typing import List, Tuple, Dict
import random
from itertools import combinations

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ROMAN = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ"]

def _letters_to_indices(gt_letters: List[str]) -> List[int]:
    """把 ['A','C'] 转成 [0,2]"""
    idx = []
    for ch in gt_letters:
        ch = ch.strip().upper()
        if ch in LETTERS:
            idx.append(LETTERS.index(ch))
    return sorted(set(idx))


def _indices_to_letters(indices: List[int]) -> List[str]:
    """把 [0,2] 转成 ['A','C']"""
    return [LETTERS[i] for i in sorted(set(indices))]


# ---------- 1) base  ----------

def make_base_variant(options: List[str],
                      gt_letters: List[str]) -> Tuple[List[str], List[str]]:
    return list(options), list(gt_letters)


# ---------- 2) shuffle  ----------

def make_shuffle_variant(options: List[str],
                         gt_letters: List[str],
                         seed: int = 0) -> Tuple[List[str], List[str], Dict]:
    """
    打乱选项，并同步打乱正确答案。
    返回:
      new_options, new_gt_letters, extra_info
    extra_info 里保存 shuffle 索引，用于调试/复现。
    """
    n = len(options)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    new_options = [options[i] for i in indices]

    # 原来的正确 index -> 新位置
    gt_idx = _letters_to_indices(gt_letters)
    new_gt_idx = []
    for gi in gt_idx:
        if gi < 0 or gi >= n:
            continue
        new_pos = indices.index(gi)
        new_gt_idx.append(new_pos)

    new_gt_letters = _indices_to_letters(new_gt_idx)

    extra = {
        "shuffle_seed": seed,
        "shuffle_indices": indices,
    }
    return new_options, new_gt_letters, extra


# ---------- 3) NOTA  ----------

def _combo_to_text(indices: List[int], correct_idx: List[int]) -> str:
    """
    把 [0,2,3] 转成 'Ⅰ、Ⅲ、Ⅳ正确' 这种描述。
    - 如果组合是正确集合 S 的真子集（非空且 A ⊂ S），前面加 'only '。
    """
    if not indices:
        return "无陈述正确"

    S = set(correct_idx)
    A = set(indices)

    romans = [ROMAN[i] for i in sorted(A)]
    base = "、".join(romans)

    # 真子集：非空，且 A ⊂ S
    if A and A < S:
        return "only " + base
    else:
        return base


def _generate_multi_nota_distractors(num_atoms: int,
                                     correct_idx: List[int],
                                     max_distractors: int = 4) -> List[List[int]]:
    """
    根据多选正确集合 correct_idx，生成若干“错误组合”，满足：
      - 类型1：真子集（不完全正确） subset(S)
      - 类型2：包含正确 + 错误混合  S ∩ A != ∅ 且 A ∩ (U-S) != ∅
      - 类型3：完全错误组合 A ⊆ (U-S)

    返回：最多 max_distractors 个组合（每个组合是 indices 列表）
    """
    all_idx = list(range(num_atoms))
    S = set(correct_idx)
    C = [i for i in all_idx if i not in S]  # complement

    combos_type1 = []  # 真子集
    combos_type2 = []  # 正确+错误混合
    combos_type3 = []  # 完全错误

    # 类型1：真子集（不等于空集，也不等于全集）
    if len(S) >= 2:
        for r in range(1, len(S)):
            for sub in combinations(S, r):
                combos_type1.append(list(sub))

    # 类型2：混合（包含正确元素 + 错误元素）
    if S and C:
        for r1 in range(1, len(S) + 1):
            for r2 in range(1, len(C) + 1):
                for sub1 in combinations(S, r1):
                    for sub2 in combinations(C, r2):
                        combos_type2.append(list(sub1 + sub2))

    # 类型3：完全错误（只从 C 里选）
    if C:
        for r in range(1, len(C) + 1):
            for sub in combinations(C, r):
                combos_type3.append(list(sub))

    # 去重 + 打散
    def normalize(lst):
        return tuple(sorted(lst))

    all_wrong_set = set()
    for lst in combos_type1 + combos_type2 + combos_type3:
        if set(lst) == S:   # 排除真正确集合
            continue
        all_wrong_set.add(normalize(lst))

    all_wrong = [list(x) for x in all_wrong_set]
    random.shuffle(all_wrong)

    # 为了尽量覆盖三类，可以按顺序抽一些
    selected: List[List[int]] = []

    def pick_from(pool):
        for combo in pool:
            if normalize(combo) in [normalize(x) for x in selected]:
                continue
            selected.append(list(combo))
            if len(selected) >= max_distractors:
                return True
        return False

    # 先尽量从三类里各选一些
    random.shuffle(combos_type1)
    random.shuffle(combos_type2)
    random.shuffle(combos_type3)

    if pick_from(combos_type1):
        return selected
    if pick_from(combos_type2):
        return selected
    if pick_from(combos_type3):
        return selected

    # 还不够就从 all_wrong 里凑满
    for combo in all_wrong:
        if normalize(combo) in [normalize(x) for x in selected]:
            continue
        selected.append(combo)
        if len(selected) >= max_distractors:
            break

    return selected


def make_nota_variant(options: List[str],
                      gt_letters: List[str],
                      nota_text: str = "以上选项均不正确 / None of the above"
                      ) -> Tuple[List[str], List[str], Dict]:
    """
    构造 NOTA 题：

    - 如果是“单选”（gt_letters 只有 1 个）：
        删除原正确选项，剩下全是错误选项，在末尾加一个 NOTA 选项。
        新正确答案 = 最后一个选项。

    - 如果是“多选”（gt_letters >= 2）：
        假设 options 表示题干中的 Ⅰ~Ⅴ 等陈述，正确集合由 gt_letters 指定。
        我们构造：
          - 前 4 个选项：错误组合（来自三类：真子集 / 混合 / 全错）
          - 第 5 个选项：NOTA（“以上组合均不正确”）
        新正确答案 = 第 5 个选项。
    """
    gt_idx = _letters_to_indices(gt_letters)
    n = len(options)

    extra: Dict = {}

    # ---------- 单选逻辑 ----------
    if len(gt_idx) == 1:
        correct_idx = gt_idx[0]
        wrong_options = [opt for i, opt in enumerate(options) if i != correct_idx]
        new_options = wrong_options + [nota_text]
        nota_idx = len(new_options) - 1
        new_gt_letters = _indices_to_letters([nota_idx])
        extra.update({
            "mode": "single_choice_nota",
            "original_correct_index": correct_idx,
            "nota_index": nota_idx,
            "nota_text": nota_text,
        })
        return new_options, new_gt_letters, extra

    # ---------- 多选逻辑 ----------
    if len(gt_idx) >= 2:
        num_atoms = min(n, len(ROMAN))  # 最多按 ROMAN 长度来
        atoms = options[:num_atoms]     # 前 num_atoms 条陈述映射为 Ⅰ~Ⅲ...等

        # 生成 4 个错误组合（用 indices 表示，例如 [0,1] 表示 Ⅰ、Ⅱ）
        distractor_idx_combos = _generate_multi_nota_distractors(
            num_atoms=num_atoms,
            correct_idx=gt_idx,
            max_distractors=4,
        )

        # 选项文本：用“Ⅰ、Ⅱ、Ⅳ正确”这种风格描述
        distractor_texts = [
            _combo_to_text(c, correct_idx=gt_idx)
            for c in distractor_idx_combos
        ]
        # 最后一个选项是 NOTA
        new_options = distractor_texts + [nota_text]

        nota_idx = len(new_options) - 1
        new_gt_letters = _indices_to_letters([nota_idx])

        extra.update({
            "mode": "multi_choice_nota",
            "num_atoms": num_atoms,
            "atoms": atoms,
            "distractor_combos": distractor_idx_combos,
            "nota_index": nota_idx,
            "nota_text": nota_text,
            "original_correct_indices": gt_idx,
        })
        return new_options, new_gt_letters, extra

    # 如果没解析出合理的 gt_idx（异常情况），退回 base
    new_options, new_gt_letters = make_base_variant(options, gt_letters)
    extra.update({"mode": "fallback_base"})
    return new_options, new_gt_letters, extra
