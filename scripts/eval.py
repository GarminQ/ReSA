from dotenv import load_dotenv
load_dotenv()

from pandora.evaluation import EvaluationData, EvaluationResult
from pandora.evaluation import PrefixEvaluator, GuardEvaluator, AgentEvaluator
from pandora.utils import load_jsonl

if __name__ == "__main__":
    eval_method = "guard"
    if eval_method == "prefix":
        evaluator = PrefixEvaluator() # 0.4980 0.2980
    elif eval_method == "guard":
        evaluator = GuardEvaluator(model_id="/home/qjm/my-model/LlamaGuard-7b") # 0.8173 0.6653 0.6480

    records = load_jsonl("/home/qjm/code/Pandora/data/baseline_results_weight-1.0.jsonl")
    # records = load_jsonl("/home/qjm/code/Pandora/data/results_weight-2.jsonl")

    data_list = [
        EvaluationData(query=record["prompt"], response=record["response"][:100])
        for record in records
    ]
    results = evaluator.evaluate(data_list)
    print(len(results))
    score = sum([item.score for item in results]) / len(results)
    print(score)


# pefix
# none 0.3076
# -0.5 0.4673
# -1.0 0.4980
# -2.0 0.5250

# guard
# none 0.8173
# -0.5 0.6730
# -1.0 0.6653
# -2.0 0.6480