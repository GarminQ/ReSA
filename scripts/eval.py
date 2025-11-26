from dotenv import load_dotenv
load_dotenv()

from pandora.evaluation import EvaluationData, EvaluationResult
from pandora.evaluation import PrefixEvaluator, GuardEvaluator, HarmEvaluator, AgentEvaluator
from pandora.utils import load_jsonl

if __name__ == "__main__":
    eval_method = "harm"

    if eval_method == "prefix":
        evaluator = PrefixEvaluator()
    elif eval_method == "guard":
        evaluator = GuardEvaluator(model_id="/home/qjm/my-model/LlamaGuard-7b")
    elif eval_method == "harm":
        evaluator = HarmEvaluator(model_id="OpenAssistant/reward-model-deberta-v3-large-v2")
    elif eval_method == "agent":
        evaluator = AgentEvaluator(model="openai/gpt-5-nano")

    # records = load_jsonl("./result/cond/baseline13b_results_weight-1.5-scaled-t1.0-full.jsonl")
    # records = load_jsonl("./result/trm/baseline7b_rm_results_weight-1.0-num10.jsonl")
    records = load_jsonl("./result/resa/13b_results_weight-1.5-num10-finish-300.jsonl")
    prompt_template = "Human: {}Assistant: "

    data_list = [
        EvaluationData(query=prompt_template.format(record["prompt"]), response=record["response"])
        for record in records
    ]
    
    results = evaluator.evaluate(data_list)
    results = [item for item in results if item.score != None]
    print(len(results))
    score = sum([item.score for item in results]) / len(results)
    print(score)




#################################################
# resa (pretopk=10)
# pefix
# none 0.3076
# -0.5 0.4673
# -1.0 0.4980
# -2.0 0.5250
# -5.0 0.6769
# -6.0 0.7634
# -7.0 0.8269
# -8.0 0.8500
# -9.0 0.8538
# -10. 0.8903
# -11. 0.8903
# -12. 0.9173
# -13. 0.9211

# guard
# none 0.8173
# -0.5 0.6730
# -1.0 0.6653
# -2.0 0.6480
# -5.0 0.6269
# -6.0 0.5480
# -7.0 0.4980
# -8.0 0.4134
# -9.0 0.4134
# -10. 0.3807
# -11. 0.3596
# -12. 0.3673
# -13. 0.3403

# agent
# -5.0 2.1237
# -8.0 2.7200
# -10. 2.6666

# new agent (:100)
# -6.0 4.1578
# -8.0 4.7634
# -10. 4.4193

# harm
# -5.0 3.4892
# -6.0 3.8856
# -7.0 4.0669
# -8.0 4.3854
# -10. 4.3987

# harm
# -8.0 4.6598

#################################################
# contrast (pretopk=10)
# prefix
# -0.5 0.6923
# -0.6 0.7519
# -0.7 0.8365
# -0.8 0.9019
# -0.9 0.9365
# -1.0 0.9576

# guard
# -0.5 0.5903
# -0.6 0.5307
# -0.7 0.4711
# -0.8 0.4211
# -0.9 0.3942
# -1.0 0.3576

# agent
# -0.8 3.5625
# new agent(:100)
# -0.8 6.3265

# harm
# -0.5 3.3617
# -0.6 3.5726
# -0.8 4.1735
# -1.0 4.5597

# harm 13b
# -0.8 4.4239
# -1.0 4.5466
# -1.5 4.6663

#################################################
# random score head
# prefix
# -5.0 0.5557
# -7.0 0.7038

# guard
# -5.0 0.6346
# -7.0 0.4865

# harm
# -5.0 3.2369
# -7.0 3.9812

#################################################
# toy(argmin 10)
# prefix 0.9807
# guard 0.2307
# agent 2.5157 (:100)
# harm 4.6312
# new agent 4.7204 (:100)

# toy(argmin 50)
# harm 4.2503

# toy(argmin 100)
# prefix 0.9980
# guard 0.1923
# harm 4.1761 ***