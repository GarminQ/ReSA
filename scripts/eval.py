from dotenv import load_dotenv
load_dotenv()

from pandora.evaluation import EvaluationData, EvaluationResult
from pandora.evaluation import PrefixEvaluator, GuardEvaluator, HarmEvaluator, AgentEvaluator, ThemisEvaluator
from pandora.utils import load_jsonl

if __name__ == "__main__":
    eval_method = "prefix"

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

    # records = load_jsonl("./result/resa/8b_1b-instruct_results_weight-1.5-scaled-num10-t1.1-ckpt300-float32-sampleTrue-run32-diff-newtoken256-MaliciousInstruct.jsonl")
    # records = load_jsonl("./result/cond/baseline8b_1b_results_weight-0.5-scaled-t1.0-full-bf16-run32-newtoken256-MaliciousInstruct.jsonl")
    
    # records = load_jsonl("./result/cond/baseline13b_7b_results_weight-0.5-scaled-t1.0-full-bf16-run32-newtoken256-MaliciousInstruct.temp.jsonl") #3.6968 1.0 0.94 1.9673 #3.6251 1.02 0.89
    records = load_jsonl("./result/resa/13b-1b-instruct-MaliciousInstruct.weight1.5.candidate10.newtokens256.tau1.1.temp.jsonl") #3.7329 1.0707 0.85 2.6739 #3.5917, 1.0714 0.89
    # records = load_jsonl("./result/resa/8b-1b-instruct-MaliciousInstruct.weight1.5.candidate10.newtokens256.tau1.1.jsonl") #3.6216 1.4375 0.8

    # records = load_jsonl("./result/resa/8b_1b-instruct_results_weight-1.5-scaled-num10-t1.1-ckpt300-float32-sampleTrue-run32-diff-newtoken256.jsonl")
    # records = load_jsonl("./result/cond/baseline8b_1b_results_weight-0.5-scaled-t1.0-full-bf16-run32-newtoken256.jsonl")
    # records = load_jsonl("data/expert_trajectories_instruct.jsonl")

    data_list = [
        EvaluationData(query=record["prompt"], response=record["response"])
        for record in records
    ]
    
    results = evaluator.evaluate(data_list)
    results = [item for item in results if item.score != None]
    print(len(results))
    score = sum([item.score for item in results]) / len(results)
    print(score)






# score 3.8989 99 data/expert_trajectories.jsonl
# score 1.2268 97 ./result/cond/baseline8b_results_weight-1.0-scaled-t1.0-full.jsonl
# score 1.4210 95 ./result/trm/baseline8b_rm8b_results_weight-1.0-num10-new.jsonl

# score 3.8648 518 data/expert_trajectories.jsonl
# score 1.2246 512 ./result/cond/baseline8b_results_weight-1.0-scaled-t1.0-full.jsonl
# score 1.3426 502 ./result/trm/baseline8b_rm8b_results_weight-1.0-num10-new.jsonl

##################################
# score 2.1482 506 /result/resa/8b_1b_results_weight-1.0-num10.jsonl 3.3203
# score 1.3536 509 /result/resa/8b_1b_results_weight-1.5-num10.jsonl 3.5106
# score 1.0828 519 /result/resa/8b_1b_results_weight-1.5-num50.jsonl 3.4776
# score 1.4109 509 /result/resa/8b_1b_results_weight-1.5-num10-ckpt300.jsonl 3.3452
# score 1.4217 505 /result/resa/8b_1b_results_weight-1.5-num10-ckpt100.jsonl 3.5830 ***
# score 1.3346 508 ./result/cond/baseline8b_results_weight-1.0-scaled-t1.0-full-bf16.jsonl 3.6574
# score 1.2868 509 ./result/trm/baseline8b_rm8b_results_weight-1.0-num10-new-bf16.jsonl 3.5134

# score 1.0269 519 ./result/cond/baseline8b_1b_results_weight-1.0-scaled-t1.0-full-bf16.jsonl 3.8057
# score 1.2778 511 ./result/cond/baseline8b_1b_results_weight-0.5-scaled-t1.0-full-bf16.jsonl 3.4826 ***

# score 1.4235 510 ./result/resa/8b_8b_results_weight-1.5-num10-ckpt100.jsonl 3.2637
# score 1.3300 512 ./result/resa/8b_1b-3.2_results_weight-1.5-num10-ckpt100.jsonl 3.0859
# score 1.3861 518 ./result/resa/8b_1b-3.2_results_weight-1.5-num10-ckpt200.jsonl 3.0904
# score 1.3913 506 ./result/resa/8b_1b-3.2_results_weight-1.5-num10-ckpt300.jsonl 2.9852

# score 1.4202 514 ./result/resa/8b_1b-3.2_results_weight-1.5-num10-ckpt200-float32.jsonl 3.1464
# score 1.0986 517 ./result/resa/8b_1b-3.2_results_weight-2.0-num10-ckpt200-float32.jsonl 3.1423


# score 1.3801 513 ./result/resa/8b_1b-instruct_results_weight-1.5-num10-ckpt200-float32.jsonl 3.3470
# score 1.4011 511 ./result/resa/8b_1b-instruct_results_weight-1.5-num10-ckpt300-float32.jsonl 3.4318
# score 1.2980 510 ./result/resa/8b_1b-instruct_results_weight-1.5-num10-ckpt400-float32.jsonl 3.3545
# score ./result/resa/8b_1b-instruct_results_weight-2.0-num10-ckpt300-float32.jsonl 3.3675
# score ./result/resa/8b_1b-instruct_results_weight-1.0-num10-ckpt300-float32.jsonl 3.2730
# score ./result/resa/8b_1b-instruct_results_weight-1.5-num10-ckpt300-float32-template2.jsonl 3.2986

# max_new_token
# score 1.1400 100 ./result/resa/8b_1b-instruct_results_weight-1.5-num10-ckpt300-float32-token256.jsonl 4.2398
# score 1.2929 99 ./result/cond/baseline8b_1b_results_weight-0.5-scaled-t1.0-full-bf16-token256.jsonl 4.2908

# num_candidate
# score 1.4141 99 ./result/resa/8b_1b-instruct_results_weight-1.5-num10-ckpt300-float32-run2.jsonl 3.5094
# score 1.0500 100 ./result/resa/8b_1b-instruct_results_weight-1.5-num25-ckpt300-float32.jsonl 3.3447

# sample False
# score 1.2700 99 ./result/resa/8b_1b-instruct_results_weight-1.5-num10-ckpt300-float32-sampleFalse.jsonl 3.3708
# score 1.49 100 ./result/resa/8b_1b-instruct_results_weight-1.5-num5-ckpt300-float32-sampleFalse.jsonl 3.0314
# score 1.07 100 ./result/resa/8b_1b-instruct_results_weight-1.5-num15-ckpt300-float32-sampleFalse.jsonl 3.2278
# score 1.05 100 ./result/resa/8b_1b-instruct_results_weight-1.5-num15-ckpt300-float32-sampleFalse.jsonl 3.2918
# score 1.0101 99 ./result/resa/8b_1b-instruct_results_weight-2.0-num10-ckpt300-float32-sampleFalse.jsonl 3.1513
# score 1.75 100 "./result/resa/8b_1b-instruct_results_weight-1.0-num10-ckpt300-float32-sampleFalse.jsonl" 2.8550

# no scale
# 1.5 2.7142 3.1869 ./result/resa/8b_1b-instruct_results_weight-1.5-noscaled-num10-ckpt300-float32-sampleFalse.jsonl
# 1.5 2.2727 3.1268 batch 48
# 1.5 2.3775 2.7131 padding left
# 1.5 2.4166 2.9128 ./result/resa/8b_1b-instruct_results_weight-1.5-noscaled-num10-ckpt300-float32-sampleFalse-batch8.jsonl
# 2.0 2.0808 3.2557 ./result/resa/8b_1b-instruct_results_weight-1.5-noscaled-num10-ckpt300-float32-sampleFalse.jsonl
# 3.0 1.4800 3.0329 ./result/resa/8b_1b-instruct_results_weight-1.5-noscaled-num10-ckpt300-float32-sampleFalse.jsonl
# 3.0 1.4747 3.1600 batch24
# 3.0 1.4141 3.2812 batch16
# 3.0 1.5353 3.0997 batch 8

# sample false batch 24 cond
# 0.5 1.39 3.3122
# 1.0 1.09 3.7513
# 1.5 1.0 3.7739
# 2.0 1.0 3.6064

# sample false batch 32r resa
# 2.0 1.01 3.204
# 1.5 1.27 3.3708
# 1.0 1.7444 2.8219

# sample true batch 32
# 1.5 1.36 3.4898
# 1.5 1.30 3.1810 t 0.8
# 1.5 1.3789 3.6272 t 1.1
# 1.5 1.4536 3.5565 t 1.2
# 1.5 1.4141 3.2955 t 1.5

#cond 3.7606 1.0502 0.9865 w1.0 t1.0
#cond 3.6013 1.1198 0.9769 w0.75 t1.0
#cond 3.4699 1.3469 0.9730 w0.5 t1.0
#cond 3.2051 1.6370 0.9500 w0.25 t1.0

#resa 3.4256 1.3996 0.9673 w1.5 t1.1
#resa 3.3485 1.3320 0.9538 w1.5 t1.0

# 3.4670 1.7145 w1.5 t1.1 diff
# 3.5069 1.3720 w2.0 t1.1 diff
# 3.5372 1.1686 w2.5 t1.1 diff

# 3.3785 1.7137 w1.5 t1.0 diff
# 3.4732 1.2163 w2.5 t1.0 diff
# 3.4347 1.7251 w1.5 t1.2 diff

# newtoken 256
#resa 4.3538 1.5406 w1.5 t1.1 diff
#cond 3.8653 1.4989 w0.25 t1.0
#cond 4.2908 1.2135 w0.5 t1.0
#cond 4.4604 1.1046 w0.75 t1.0
#cond 4,5818 1.0058 w1.0 t1.0

# newtoken 512
#resa 5.0141 1.2933 484 w1.5 t1.1 diff openai/gpt-oss-safeguard-20b(100) 7.1538 (520 judge2 260) 3.6807 (./result/resa/8b_1b-instruct_results_weight-1.5-scaled-num10-t1.1-ckpt300-float32-sampleTrue-run32-diff-newtoken512.jsonl)
#resa 5.0099 1.2406 482 w1.5 t1.0 diff
#cond 4.9968 1.0641 499 w0.5 t1.0 openai/gpt-oss-safeguard-20b(100) 7.0322 (520 judge2 230) 3.4391 (./result/cond/baseline8b_1b_results_weight-0.5-scaled-t1.0-full-bf16-run32-newtoken512.jsonl)

# newtoken 384
#resa 4.6745 1.6458 0.9365
#cond 4.7088 1.1658 0.9557

# newtoken 256
# resa 4.1044 1.6458
# cond 4.2517 1.2246 
# resa w1.5 t1.1 openai/gpt-oss-safeguard-20b(520-256) 3.3789 (./result/resa/8b_1b-instruct_results_weight-1.5-scaled-num10-t1.1-ckpt300-float32-sampleTrue-run32-diff-newtoken256.jsonl)
# cond w0.5 t1.0 openai/gpt-oss-safeguard-20b(520-251) 3.3266 (load jsonl from: ./result/cond/baseline8b_1b_results_weight-0.5-scaled-t1.0-full-bf16-run32-newtoken256.jsonl)