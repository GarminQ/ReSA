from dotenv import load_dotenv
load_dotenv()

import torch
from pandora.evaluation import EvaluationData, EvaluationResult
from pandora.evaluation import PrefixEvaluator, GuardEvaluator, HarmEvaluator, AgentEvaluator, ThemisEvaluator
from pandora.utils import load_jsonl
1
if __name__ == "__main__":
    eval_methods = ["prefix", "harm", "themis"]
    
    eval_json_path = "output/result/wtos/Qwen2.5-7B-Instruct_Qwen2.5-3B-Instruct_Qwen2.5-3B/AdvBench_w4.0_new256_tau1.0_topp1.0-25.jsonl"

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
            evaluator = ThemisEvaluator(model_id="/root/autodl-tmp/my-model/Themis")

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


# =============================
# Available item count: 520
# prefix score: 0.6653846153846154
# Load jsonl from: output/cond/gemma-7b-it-hf_gemma-2b-it_gemma-2b/AdvBench_w0.5_c10_new256_tau1.0.jsonl
# Available item count: 520
# harm score: -1.5619672335111179
# Available item count: 514
# themis score: 2.28988326848249

# Available item count: 520
# prefix score: 0.6788461538461539
# Load jsonl from: result/resa/gemma-7b-it_Llama-3.2-1B-Instruct/AdvBench_w1.50_c10_new256_tau1.1.jsonl
# Available item count: 520
# harm score: -1.6122467517852783
# Available item count: 508
# themis score: 3.173228346456693

# Load jsonl from: result/cond/gemma-7b-it_gemma-2b-it_gemma-2b/HarmBench_w0.5_c10_new256_tau1.0.jsonl
# Available item count: 200
# prefix score: 0.76
# Load jsonl from: result/cond/gemma-7b-it_gemma-2b-it_gemma-2b/HarmBench_w0.5_c10_new256_tau1.0.jsonl
# Available item count: 200
# harm score: -1.550948486328125
# Available item count: 197
# themis score: 2.269035532994924

# Load jsonl from: result/resa/gemma-7b-it_Llama-3.2-1B-Instruct/HarmBench_w1.50_c10_new256_tau1.1.jsonl
# Available item count: 200
# prefix score: 0.795
# Load jsonl from: result/resa/gemma-7b-it_Llama-3.2-1B-Instruct/HarmBench_w1.50_c10_new256_tau1.1.jsonl
# Available item count: 200
# harm score: -1.7524365234375
# Available item count: 194
# themis score: 2.479381443298969

#===============================

