from dotenv import load_dotenv
load_dotenv()

from pandora.evaluation import EvaluationData, EvaluationResult
from pandora.evaluation import PrefixEvaluator, GuardEvaluator, HarmEvaluator, AgentEvaluator, ThemisEvaluator
from pandora.utils import load_jsonl

if __name__ == "__main__":
    eval_method = "prefix"
    eval_json_path = "output/resa/Llama-3.1-8B-Instruct_Llama-3.2-1B-Instruct/AdvBench_w1.50_c10_new256_tau1.1.jsonl"
    
    if eval_method == "prefix":
        evaluator = PrefixEvaluator()
    elif eval_method == "guard":
        evaluator = GuardEvaluator(model_id="/home/qjm/my-model/Llama-Guard-3-8B") # /home/qjm/my-model/LlamaGuard-7b
    elif eval_method == "harm":
        evaluator = HarmEvaluator(model_id="OpenAssistant/reward-model-deberta-v3-large-v2")
    elif eval_method == "agent":
        evaluator = AgentEvaluator(model="openai/gpt-oss-120b")
    elif eval_method == "themis":
        evaluator = ThemisEvaluator(model_id="/home/qjm/my-model/Themis")

    records = load_jsonl(eval_json_path)
    data_list = [
        EvaluationData(query=record["prompt"], response=record["response"])
        for record in records
    ]

    results = evaluator.evaluate(data_list)
    results = [item for item in results if item.score != None]
    print(f"Available item count: {len(results)}")
    score = sum([item.score for item in results]) / len(results)
    print(f"{eval_method} score: {score:.2}")