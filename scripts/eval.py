from dotenv import load_dotenv
load_dotenv()

from pandora.evaluation import EvaluationData, EvaluationResult
from pandora.evaluation import PrefixEvaluator, GuardEvaluator, HarmEvaluator, AgentEvaluator, ThemisEvaluator
from pandora.utils import load_jsonl

if __name__ == "__main__":
    eval_method = "themis"

    if eval_method == "prefix":
        evaluator = PrefixEvaluator()
    elif eval_method == "guard":
        evaluator = GuardEvaluator(model_id="/home/qjm/my-model/LlamaGuard-7b")
    elif eval_method == "harm":
        evaluator = HarmEvaluator(model_id="OpenAssistant/reward-model-deberta-v3-large-v2")
    elif eval_method == "agent":
        evaluator = AgentEvaluator(model="openai/gpt-5-nano")
    elif eval_method == "themis":
        evaluator = ThemisEvaluator(model_id="/home/qjm/my-model/Themis")

    records = load_jsonl("data/expert_trajectories.jsonl")
    # records = load_jsonl("./result/trm/baseline8b_rm8b_results_weight-1.0-num10-new.jsonl")
    # records = load_jsonl("./result/resa/13b_results_weight-1.0-num10-finish-300.jsonl")

    data_list = [
        EvaluationData(query=record["prompt"], response=record["response"])
        for record in records
    ]
    
    results = evaluator.evaluate(data_list)
    results = [item for item in results if item.score != None]
    print(len(results))
    score = sum([item.score for item in results]) / len(results)
    print(score)
