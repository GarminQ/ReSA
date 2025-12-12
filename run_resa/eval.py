from dotenv import load_dotenv
load_dotenv()

import torch
from pandora.evaluation import EvaluationData, EvaluationResult
from pandora.evaluation import PrefixEvaluator, GuardEvaluator, HarmEvaluator, AgentEvaluator, ThemisEvaluator, PPLEvaluator
from pandora.utils import load_jsonl

if __name__ == "__main__":
    eval_methods = ["prefix", "harm", "guard", "ppl", "themis"]
    eval_json_path = "output/result/resa/Meta-Llama-3.1-70B-Instruct-AWQ-INT4_Llama-3.2-1B-Instruct/HarmBench_w1.0_c10_new256_tau1.0-run2.jsonl"

    for eval_method in eval_methods:
        if eval_method == "prefix":
            evaluator = PrefixEvaluator()
        elif eval_method == "guard":
            evaluator = GuardEvaluator(model_id="/root/autodl-tmp/my-model/Llama-Guard-3-8B") # /home/qjm/my-model/LlamaGuard-7b
        elif eval_method == "harm":
            evaluator = HarmEvaluator(model_id="/root/autodl-tmp/my-model/reward-model-deberta-v3-large-v2")
        elif eval_method == "agent":
            evaluator = AgentEvaluator(model="openai/gpt-oss-120b")
        elif eval_method == "themis":
            evaluator = ThemisEvaluator(model_id="/root/autodl-tmp/my-model/Themis")
        elif eval_method == "ppl":
            evaluator = PPLEvaluator(model_id="/root/autodl-tmp/my-model/gpt2-xl")

        records = load_jsonl(eval_json_path)
        data_list = [
            EvaluationData(query=record["prompt"], response=record["response"])
            for record in records
        ]
        if eval_method in ["prefix", "agent"]:
            results = evaluator.evaluate(data_list)
        else:
            batch_size= 32 if eval_method == "themis" else 4
            results = evaluator.evaluate(data_list, batch_size=batch_size)
        results = [item for item in results if item.score != None]
        print(f"Available item count: {len(results)}")
        score = sum([item.score for item in results]) / len(results)
        print(f"{eval_method} score: {score}")

        del evaluator
        torch.cuda.empty_cache()

