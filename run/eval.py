import torch
from pandora.evaluation import EvaluationData, EvaluationResult
from pandora.evaluation import PrefixEvaluator, HarmEvaluator, AgentEvaluator, PPLEvaluator
from pandora.utils import load_jsonl

if __name__ == "__main__":
    eval_methods = ["prefix", "harm", "agent", "ppl"]
    eval_json_path = "your_json_path"

    for eval_method in eval_methods:
        if eval_method == "prefix":
            evaluator = PrefixEvaluator()
        elif eval_method == "harm":
            evaluator = HarmEvaluator(model_id="OpenAssistant/reward-model-deberta-v3-large-v2")
        elif eval_method == "agent":
            evaluator = AgentEvaluator(model="openai/gpt-oss-120b")
        elif eval_method == "ppl":
            evaluator = PPLEvaluator(model_id="deepseek-ai/DeepSeek-V2-Lite")

        records = load_jsonl(eval_json_path)
        data_list = [
            EvaluationData(query=record["prompt"], response=record["response"])
            for record in records
        ]

        results = evaluator.evaluate(data_list)
        results = [item for item in results if item.score != None]
        print(f"Available item count: {len(results)}")
        score = sum([item.score for item in results]) / len(results)
        print(f"{eval_method} score: {score}")

        del evaluator
        torch.cuda.empty_cache()

