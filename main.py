import argparse
from .config import OpenAIConfig
from .clients import OpenAIClient
from .data import load_dataset
from .judge import RuleJudge, LLMJudge
from .eval import run_eval
from .utils import save_json, save_csv

def main():
    ap = argparse.ArgumentParser("Medical LLM Evaluation Demo")
    ap.add_argument("--data", required=True, help="评测数据集 JSON 路径")
    ap.add_argument("--out_json", default="results.json")
    ap.add_argument("--out_csv", default="records.csv")
    ap.add_argument("--use_llm_judge", action="store_true",
                    help="open_response 是否使用 gpt-4o 裁判")
    args = ap.parse_args()

    cfg = OpenAIConfig()
    client = OpenAIClient(
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        default_model=cfg.test_model,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
    )

    # 裁判模型：如果 use_llm_judge，则同一个 client 但指定 judge_model=gpt-4o
    if args.use_llm_judge:
        judge = LLMJudge(client, judge_model=cfg.judge_model)
    else:
        judge = RuleJudge()

    ds = load_dataset(args.data)
    result = run_eval(ds, client, judge, test_model=cfg.test_model)

    save_json(result, args.out_json)
    save_csv(result["records"], args.out_csv)
    print("Summary:", result["summary"])

if __name__ == "__main__":
    main()
