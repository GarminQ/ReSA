from dotenv import load_dotenv
load_dotenv()

import torch
from pandora.evaluation import EvaluationData, EvaluationResult
from pandora.evaluation import PrefixEvaluator, GuardEvaluator, HarmEvaluator, AgentEvaluator, ThemisEvaluator
from pandora.utils import load_jsonl

if __name__ == "__main__":
    eval_methods = ["prefix", "harm"]
    eval_json_path = "output/cond/Llama-2-7b-hf_gemma-2b-it_gemma-2b/AdvBench_w0.5_c10_new256_tau1.0.jsonl"

    for eval_method in eval_methods:
        if eval_method == "prefix":
            evaluator = PrefixEvaluator()
        elif eval_method == "guard":
            evaluator = GuardEvaluator(model_id="/home/qjm/my-model/Llama-Guard-3-8B") # /home/qjm/my-model/LlamaGuard-7b
        elif eval_method == "harm":
            evaluator = HarmEvaluator(model_id="/root/autodl-tmp/my-model/reward-model-deberta-v3-large-v2")
        elif eval_method == "agent":
            evaluator = AgentEvaluator(model="openai/gpt-oss-120b")
        elif eval_method == "themis":
            evaluator = ThemisEvaluator(model_id="/home/qjm/my-model/Themis")

        records = load_jsonl(eval_json_path)
        data_list = [
            EvaluationData(query=record["prompt"], response=record["response"])
            for record in records
        ]
        if eval_method in ["prefix", "agent"]:
            results = evaluator.evaluate(data_list)
        else:
            results = evaluator.evaluate(data_list, batch_size=12)
        results = [item for item in results if item.score != None]
        print(f"Available item count: {len(results)}")
        score = sum([item.score for item in results]) / len(results)
        print(f"{eval_method} score: {score}")

        del evaluator
        torch.cuda.empty_cache()


# Available item count: 497
# themis score: 1.6  
# Available item count: 520
# harm score: -4.2
# Available item count: 520
# prefix score: 0.96

# Available item count: 487
# themis score: 2.0164
# Available item count: 520
# harm score: -3.2
# Available item count: 520
# prefix score: 0.9

# Available item count: 520
# prefix score: 0.6653846153846154
# Load jsonl from: output/cond/Llama-2-7b-hf_gemma-2b-it_gemma-2b/AdvBench_w0.5_c10_new256_tau1.0.jsonl
# Available item count: 520
# harm score: -1.5619672335111179