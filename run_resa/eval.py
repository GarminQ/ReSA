from dotenv import load_dotenv
load_dotenv()

import torch
from pandora.evaluation import EvaluationData, EvaluationResult
from pandora.evaluation import PrefixEvaluator, GuardEvaluator, HarmEvaluator, AgentEvaluator, ThemisEvaluator, PPLEvaluator
from pandora.utils import load_jsonl

if __name__ == "__main__":
    # eval_methods = ["prefix", "agent"]
    eval_methods = ["prefix", "harm", "guard", "ppl", "themis"]
    # eval_json_path = "output/result/cond/gemma-2-27b-it_gemma-2b-it_gemma-2b/HarmBench_w0.3_new256_tau1.0_topp1.0.jsonl"
    # eval_json_path = "output/result/resa/Qwen3-14B_Llama-3.2-1B-Instruct/HarmBench_w2.5_c10_new256_tau1.0.jsonl"

    # eval_json_path = "output/result/cond/Qwen3-14B_Qwen2.5-3B-Instruct_Qwen2.5-3B/AdvBench_w0.6_new256_tau1.0_topp1.0.jsonl"
    # eval_json_path = "output/result/wtos/Qwen3-14B_Qwen2.5-3B-Instruct_Qwen2.5-3B/AdvBench_w3.0_new256_tau1.0_topp1.0-25.jsonl"
    # eval_json_path = "output/result/resa/Qwen3-14B_Llama-3.2-1B-Instruct/AdvBench_w1.5_c10_new256_tau1.0.jsonl"

    # eval_json_path = "output/result/cond/gemma-2-27b-it_gemma-2b-it_gemma-2b/HarmBench_w0.3_new256_tau1.0_topp1.0.jsonl"
    # eval_json_path = "output/result/wtos/gemma-2-27b-it_gemma-2b-it_gemma-2b/HarmBench_w1.25_new256_tau1.0_topp1.0-125.jsonl"
    # eval_json_path = "output/result/resa/gemma-2-27b-it_Llama-3.2-1B-Instruct/HarmBench_w1.5_c10_new256_tau1.0.jsonl"

    # eval_json_path = "output/result/cond/Meta-Llama-3.1-70B-Instruct-AWQ-INT4_Llama-3.2-3B-Instruct_Llama-3.2-3B/HarmBench_w0.2_new256_tau1.0_topp1.0.jsonl"
    # eval_json_path = "output/result/wtos/Meta-Llama-3.1-70B-Instruct-AWQ-INT4_Llama-3.2-3B-Instruct_Llama-3.2-3B/HarmBench_w0.5_new256_tau1.0_topp1.0-25.jsonl"
    # eval_json_path = "output/result/resa/Meta-Llama-3.1-70B-Instruct-AWQ-INT4_Llama-3.2-1B-Instruct/HarmBench_w1.25_c10_new256_tau1.0.jsonl"

    for eval_method in eval_methods:
        if eval_method == "prefix":
            evaluator = PrefixEvaluator()
        elif eval_method == "guard":
            evaluator = GuardEvaluator(model_id="/root/autodl-tmp/my-model/Llama-Guard-3-8B") # /home/qjm/my-model/LlamaGuard-7b
        elif eval_method == "harm":
            evaluator = HarmEvaluator(model_id="/root/autodl-tmp/my-model/reward-model-deberta-v3-large-v2")
        elif eval_method == "agent":
            evaluator = AgentEvaluator(model="openai/gpt-oss-120b") # openai/gpt-oss-120b 
        elif eval_method == "themis":
            evaluator = ThemisEvaluator(model_id="/root/autodl-tmp/my-model/Themis")
        elif eval_method == "ppl":
            evaluator = PPLEvaluator(model_id="/root/autodl-tmp/my-model/Llama-3.1-8B")

        records = load_jsonl(eval_json_path)
        data_list = [
            EvaluationData(query=record["prompt"], response=record["response"])
            for record in records
        ]
        if eval_method in ["prefix", "agent"]:
            results = evaluator.evaluate(data_list)
        else:
            batch_size= 32 if eval_method == "themis" else 16
            results = evaluator.evaluate(data_list, batch_size=batch_size)
        results = [item for item in results if item.score != None]
        print(f"Available item count: {len(results)}")
        score = sum([item.score for item in results]) / len(results)
        print(f"{eval_method} score: {score}")

        del evaluator
        torch.cuda.empty_cache()

