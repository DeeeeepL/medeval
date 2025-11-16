import argparse
from pathlib import Path
import sys
import os

# ========= 关键 hack：支持 python main.py =========
if __name__ == "__main__" and __package__ is None:
    # 当前文件： .../评测/medeval/main.py
    # 我们要把上一级目录（.../评测）加入 sys.path，并把 __package__ 设为 "medeval"
    this_file = os.path.abspath(__file__)
    pkg_dir = os.path.dirname(this_file)            # .../评测/medeval
    parent_dir = os.path.dirname(pkg_dir)           # .../评测
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    __package__ = "medeval"
# ==================================================

# 下面就可以放心用包内相对导入了
from config import load_eval_config
from clients import OpenAIClient
from data import load_dataset
from judge import RuleJudge, LLMJudge
from eval.evaluator import run_eval
from utils import save_json, save_csv


def main():
    os.environ['http_proxy'] = 'http://127.0.0.1:8001'
    os.environ['https_proxy'] = 'http://127.0.0.1:8001'

    ap = argparse.ArgumentParser("Medical LLM Evaluation Demo")
    ap.add_argument(
        "--data",
        nargs="+",
        required=False,
        # default=["dataset/问答题示例.json",
        #          "dataset/选择题示例.json",],
        default=["dataset/选择题示例.json"],

        help="一个或多个评测数据集 JSON 路径"
    )
    ap.add_argument(
        "--out_dir",
        default="results",
        help="结果输出目录，每个数据集单独生成 json/csv"
    )
    ap.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="open_response 是否使用裁判模型 (gpt-4o)；否则用规则裁判"
    )
    ap.add_argument(
        "--choice_modes",
        nargs="+",
        default=["all"],
        choices=["base", "shuffle", "nota", "all"],
        help="选择题评测模式：base / shuffle / nota / all"
    )

    args = ap.parse_args()

    choice_modes = args.choice_modes
    if "all" in choice_modes:
        choice_modes = ["base", "shuffle", "nota"]

    cfg = load_eval_config()

    # 1️⃣ 待测模型 
    test_client = OpenAIClient(
        api_base=cfg.test.api_base,
        api_key=cfg.test.api_key,
        default_model=cfg.test.model,
        temperature=cfg.test.temperature,
        timeout=cfg.test.timeout,
    )

    # 2️⃣ 裁判模型 client（比如 gpt-4o）
    judge_client = OpenAIClient(
        api_base=cfg.judge.api_base,
        api_key=cfg.judge.api_key,
        default_model=cfg.judge.model,
        temperature=cfg.judge.temperature,
        timeout=cfg.judge.timeout,
    )

    # 3️⃣ 选择裁判实现
    if args.use_llm_judge:
        judge = LLMJudge(judge_client)   # GPT-4o 按 scoring points 给 flag
    else:
        judge = RuleJudge()              # 简单规则裁判（子串匹配）

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4️⃣ 多个数据集：逐个评测、分别输出结果文件
    for data_path in args.data:
        ds = load_dataset(data_path)
        res = run_eval(
            ds,
            test_client,
            judge,
            test_model=cfg.test.model,
            choice_modes=choice_modes, 
        )

        ds_id = ds.dataset_metadata.dataset_id
        ds_name = ds.dataset_metadata.dataset_name

        base = f"{ds_id}__{cfg.test.model}"
        json_path = out_dir / f"{base}.json"
        csv_path = out_dir / f"{base}.csv"

        save_json(res, json_path)
        save_csv(res["records"], csv_path)

        print(f"[DONE] Dataset: {ds_id} ({ds_name})")
        print(f"       -> {json_path}")
        print(f"       -> {csv_path}")


if __name__ == "__main__":
    main()
