from dotenv import load_dotenv
load_dotenv()

import json
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (  
    HfArgumentParser, 
    AutoModelForCausalLM,   
    AutoModelForSequenceClassification,  
)  

from pandora.generation import RewardGuidedGenerator
from pandora.models import CustomRewardModel
from pandora.arguments import RSGenerationConfig, ModelArguments, DataArguments
from pandora.utils import prepare_tokenizer, load_jsonl

if __name__ == "__main__":
    parser = HfArgumentParser((RSGenerationConfig, ModelArguments, DataArguments))
    generation_args, model_args, data_args = parser.parse_args_into_dataclasses()
    model_args.policy_model_id = "/home/qjm/my-model/Llama-2-7b-chat-hf"

    policy_model = AutoModelForCausalLM.from_pretrained(  
        model_args.policy_model_id,   
        torch_dtype=torch.bfloat16,   
        device_map='auto'  
    )  

    reward_model = CustomRewardModel.from_pretrained_backbone(  
        model_args.reward_model_id,
        torch_dtype=torch.float32,   
        device_map='auto'  
    ) 

    score_head_path = "output/policy_output_new/checkpoint-200/reward_model/score_head.pt"
    score_head_state = torch.load(score_head_path, map_location="cpu", weights_only=True)  
    reward_model.score_head.load_state_dict(score_head_state)  
    print(f"Loaded score_head from {score_head_path}") 

    policy_tokenizer = prepare_tokenizer(model_args.policy_model_id)
    reward_tokenizer = prepare_tokenizer(model_args.reward_model_id)

    generator = RewardGuidedGenerator(
        policy_model=policy_model, 
        reward_model=reward_model, 
        policy_tokenizer=policy_tokenizer, 
        reward_tokenizer=reward_tokenizer, 
        device="cuda")  
    
    json_data = load_jsonl("./data/expert_trajectories.jsonl")
    prompt_data = [item['prompt'] for item in json_data]
    batch_size = 1
    with open("./data/results_weight-5.0.jsonl", "w") as f:
        for i in tqdm(range(0, len(json_data), batch_size)):
            batch_prompt = prompt_data[i:i+batch_size]
            batch_output = generator.generate(prompts=batch_prompt, num_candidate_tokens=100, 
                                              max_new_tokens=128, reward_weight=-5.0, temperature=1.0)  
            for prompt, output in zip(batch_prompt, batch_output):
                f.write(json.dumps({"prompt": prompt, "response": output}, ensure_ascii=False) + "\n")
