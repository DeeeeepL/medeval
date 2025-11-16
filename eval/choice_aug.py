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

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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
    """
    原始题目，不做增强
    """
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

def make_nota_variant(options: List[str],
                      gt_letters: List[str],
                      nota_text: str = "以上选项均不正确 / None of the above") -> Tuple[List[str], List[str], Dict]:
    """
    构造 NOTA 题（仅对单选题：len(gt_letters) == 1 时生效）：
    - 删除原正确选项
    - 在末尾新增一个 NOTA 选项
    - 正确答案变为最后一个选项

    如果不是单选（例如多选），为了安全起见，直接退回 base。
    """
    gt_idx = _letters_to_indices(gt_letters)
    if len(gt_idx) != 1:
        # 多选暂时不做 NOTA 增强，直接退回原题
        return make_base_variant(options, gt_letters) + ({},)  # 补一个 extra

    correct_idx = gt_idx[0]

    wrong_options = [opt for i, opt in enumerate(options) if i != correct_idx]
    # 新增 NOTA 选项
    nota_option = nota_text
    new_options = wrong_options + [nota_option]

    nota_idx = len(new_options) - 1
    new_gt_letters = _indices_to_letters([nota_idx])

    extra = {
        "original_correct_index": correct_idx,
        "nota_index": nota_idx,
        "nota_text": nota_text,
    }
    return new_options, new_gt_letters, extra
