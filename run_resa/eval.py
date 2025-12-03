from dotenv import load_dotenv
load_dotenv()

import torch
from pandora.evaluation import EvaluationData, EvaluationResult
from pandora.evaluation import PrefixEvaluator, GuardEvaluator, HarmEvaluator, AgentEvaluator, ThemisEvaluator
from pandora.utils import load_jsonl

if __name__ == "__main__":
    eval_methods = ["prefix", "harm", "themis"]

    # eval_json_path = "result/cond/Llama-2-7b-hf_gemma-2b-it_gemma-2b/AdvBench_w0.5_c10_new256_tau1.0.jsonl"
    # eval_json_path = "result/resa/gemma-7b-it_Llama-3.2-1B-Instruct/AdvBench_w1.50_c10_new256_tau1.1.jsonl"
    # eval_json_path = "result/cond/Qwen2.5-7B-Instruct_Qwen2.5-3B-Instruct_Qwen2.5-3B/HarmBench_w0.75_c10_new256_tau1.0.jsonl" # agent
    # eval_json_path = "result/resa/Qwen2.5-7B-Instruct_Llama-3.2-1B-Instruct/HarmBench_w2.00_c10_new256_tau1.1.jsonl" # agent
    eval_json_path = "result/resa/Qwen2.5-7B-Instruct_Llama-3.2-1B-Instruct/AdvBench_w3.0_c10_new256_tau1.1.jsonl"
    # eval_json_path = "result/resa/Qwen2.5-7B-Instruct_Llama-3.2-1B-Instruct/AdvBench_w2.0_c10_new256_tau1.1.jsonl"
    # eval_json_path = "result/cond/Qwen2.5-7B-Instruct_Qwen2.5-3B-Instruct_Qwen2.5-3B/AdvBench_w0.75_new256_tau1.0_topp1.0.jsonl"

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
            batch_size= 16 if eval_method == "themis" else 4
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
# Load jsonl from: output/cond/Llama-2-7b-hf_gemma-2b-it_gemma-2b/AdvBench_w0.5_c10_new256_tau1.0.jsonl
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

# Load jsonl from: result/cond/Llama-2-7b-hf_gemma-2b-it_gemma-2b/HarmBench_w0.5_c10_new256_tau1.0.jsonl
# Available item count: 200
# prefix score: 0.76
# Load jsonl from: result/cond/Llama-2-7b-hf_gemma-2b-it_gemma-2b/HarmBench_w0.5_c10_new256_tau1.0.jsonl
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
# Available item count: 200
# prefix score: 0.275
# Load jsonl from: result/cond/Qwen2.5-7B-Instruct_Qwen2.5-3B-Instruct_Qwen2.5-3B/HarmBench_w0.5_c10_new256_tau1.0.jsonl
# Available item count: 200
# harm score: -1.589002685546875
# Load jsonl from: result/cond/Qwen2.5-7B-Instruct_Qwen2.5-3B-Instruct_Qwen2.5-3B/HarmBench_w0.5_c10_new256_tau1.0.jsonl
# Available item count: 191
# themis score: 3.93717277486911

# Available item count: 200
# prefix score: 0.47
# Load jsonl from: result/resa/Qwen2.5-7B-Instruct_Llama-3.2-1B-Instruct/HarmBench_w1.50_c10_new256_tau1.1.jsonl
# Available item count: 200
# harm score: -2.26791015625
# Load jsonl from: result/resa/Qwen2.5-7B-Instruct_Llama-3.2-1B-Instruct/HarmBench_w1.50_c10_new256_tau1.1.jsonl
# Available item count: 171
# themis score: 2.6608187134502925

# Available item count: 200
# prefix score: 0.925
# Load jsonl from: result/cond/Qwen2.5-7B-Instruct_Qwen2.5-3B-Instruct_Qwen2.5-3B/HarmBench_w1.0_c10_new256_tau1.0.jsonl
# Available item count: 200
# harm score: -3.846064453125
# Load jsonl from: result/cond/Qwen2.5-7B-Instruct_Qwen2.5-3B-Instruct_Qwen2.5-3B/HarmBench_w1.0_c10_new256_tau1.0.jsonl
# Available item count: 196
# themis score: 1.086734693877551

# Available item count: 200
# prefix score: 0.55
# Load jsonl from: result/resa/Qwen2.5-7B-Instruct_Llama-3.2-1B-Instruct/HarmBench_w2.00_c10_new256_tau1.1.jsonl
# Available item count: 200
# harm score: -2.519296875
# Load jsonl from: result/resa/Qwen2.5-7B-Instruct_Llama-3.2-1B-Instruct/HarmBench_w2.00_c10_new256_tau1.1.jsonl
# Available item count: 177
# themis score: 1.9039548022598871
# Available item count: 191
# agent score: 1.7905759162303665

# Load jsonl from: result/cond/Qwen2.5-7B-Instruct_Qwen2.5-3B-Instruct_Qwen2.5-3B/HarmBench_w0.75_c10_new256_tau1.0.jsonl
# Available item count: 200
# prefix score: 0.67
# Available item count: 200
# harm score: -2.855828857421875
# Available item count: 192
# themis score: 1.4791666666666667
# Available item count: 187
# agent score: 2.2887700534759357


# Load jsonl from: result/resa/Qwen2.5-7B-Instruct_Llama-3.2-1B-Instruct/AdvBench_w2.0_c10_new256_tau1.1.jsonl
# Available item count: 520
# prefix score: 0.49615384615384617
# Available item count: 520
# harm score: -1.9730243389423077
# Available item count: 494
# themis score: 2.5121457489878543

# Load jsonl from: result/resa/Qwen2.5-7B-Instruct_Llama-3.2-1B-Instruct/AdvBench_w3.0_c10_new256_tau1.1.jsonl
# Available item count: 520
# prefix score: 0.5346153846153846
# Available item count: 520
# harm score: -2.1505220853365383
# Available item count: 504
# themis score: 1.5476190476190477

# Load jsonl from: result/cond/Qwen2.5-7B-Instruct_Qwen2.5-3B-Instruct_Qwen2.5-3B/AdvBench_w0.75_new256_tau1.0_topp1.0.jsonl
# Available item count: 520
# prefix score: 0.5903846153846154
# Available item count: 520
# harm score: -2.8219515286959136
# Available item count: 487
# themis score: 1.5195071868583163

