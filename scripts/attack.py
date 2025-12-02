from dotenv import load_dotenv
load_dotenv()

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import (  
    HfArgumentParser, 
    AutoTokenizer, 
    AutoModelForCausalLM
)  

from pandora.generation import RewardGuidedGenerator
from pandora.models import CustomRewardModel  
from pandora.arguments import RSGenerationConfig, ModelArguments, DataArguments
from pandora.utils import get_quantization_config, prepare_tokenizer, reward_chat_template, get_eval_data

if __name__ == "__main__":
    parser = HfArgumentParser((RSGenerationConfig, ModelArguments, DataArguments))
    gen_args, model_args, data_args = parser.parse_args_into_dataclasses()

    quantization_config = get_quantization_config(model_args.quantization)
    print(Path(model_args.model_base_path) / model_args.policy_model_id)
    policy_model = AutoModelForCausalLM.from_pretrained(  
        Path(model_args.model_base_path) / model_args.policy_model_id,   
        quantization_config=quantization_config, 
        dtype=torch.float16 if quantization_config is None else None, 
        device_map='auto'  
    )  

    reward_model = CustomRewardModel.from_pretrained_backbone(  
        Path(model_args.model_base_path) / model_args.reward_model_id,
        pooling_mode="last", 
        dtype=torch.float32,   
        device_map='auto'  
    ) 

    reward_head_state = torch.load(model_args.reward_head_path, map_location="cpu", weights_only=True)  
    reward_model.score_head.load_state_dict(reward_head_state)  
    print(f"Load score_head from: {model_args.reward_head_path}") 

    policy_tokenizer = prepare_tokenizer(Path(model_args.model_base_path) / model_args.policy_model_id)
    reward_tokenizer = AutoTokenizer.from_pretrained(Path(model_args.model_base_path) / model_args.reward_model_id)
    reward_tokenizer.chat_template = reward_chat_template

    generator = RewardGuidedGenerator(
        policy_model=policy_model, 
        reward_model=reward_model, 
        policy_tokenizer=policy_tokenizer, 
        reward_tokenizer=reward_tokenizer
    )  
    prompt_data = get_eval_data(data_args.attack_dataset_name)
    
    result_base_path = Path(gen_args.result_base_path)
    save_result_path = (
        result_base_path / f"{model_args.policy_model_id}_{model_args.reward_model_id}" /
        f"{data_args.attack_dataset_name}_w{gen_args.reward_weight:.2f}_c{gen_args.num_candidate_tokens}_new{gen_args.max_new_tokens}_tau{gen_args.temperature}.jsonl"
    )
    save_result_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Save attack result: {save_result_path}")

    with open(save_result_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(prompt_data), gen_args.batch_size), desc="Genrating"):
            batch_prompt = prompt_data[i:i+gen_args.batch_size]
            batch_output = generator.generate(
                                prompts=batch_prompt, 
                                num_candidate_tokens=gen_args.num_candidate_tokens, 
                                max_new_tokens=gen_args.max_new_tokens, 
                                reward_weight=-gen_args.reward_weight, 
                                temperature=gen_args.temperature, 
                                do_sample=gen_args.do_sample
                                )  
            for prompt, output in zip(batch_prompt, batch_output):
                f.write(json.dumps({"prompt": prompt, "response": output}, ensure_ascii=False) + "\n")
