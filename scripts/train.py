from dotenv import load_dotenv
load_dotenv()

import torch
from pathlib import Path
from datasets import Dataset, load_dataset  
from peft import LoraConfig  
from transformers import HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from pandora.trainer import MaxEntIRLTrainer
from pandora.models import CustomRewardModel
from pandora.arguments import MaxEntIRLConfig, ModelArguments, DataArguments
from pandora.utils import prepare_tokenizer, base_prompt_template, reward_chat_template, get_train_data


if __name__ == "__main__":
    parser = HfArgumentParser((MaxEntIRLConfig, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    model_args.policy_model_id = "/home/qjm/my-model/Llama-3.2-1B"
    model_args.reward_model_id = "/home/qjm/my-model/Llama-3.2-1B-Instruct"
    policy_model = AutoModelForCausalLM.from_pretrained(  
        model_args.policy_model_id,   
        torch_dtype=torch.bfloat16,
        # quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map='auto'  
    )  
    reward_model = CustomRewardModel.from_pretrained_backbone(  
        model_args.reward_model_id,
        pooling_mode="mean", 
        # torch_dtype=torch.float32, 
        torch_dtype=torch.bfloat16, 
        # quantization_conbfig=BitsAndBytesConfig(load_in_4bit=True),
        device_map='auto' 
    ) 
    peft_config = LoraConfig(  
        r=model_args.lora_r,  
        lora_alpha=model_args.lora_alpha,  
        lora_dropout=model_args.lora_dropout,  
        bias="none",  
        task_type="CAUSAL_LM",  
    )  
    reward_tokenizer = AutoTokenizer.from_pretrained(model_args.reward_model_id)
    reward_tokenizer.chat_template = reward_chat_template
    if reward_tokenizer.pad_token is None:    
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    prompt_dataset = get_train_data(data_args.prompt_dataset_name)
    prompt_dataset = prompt_dataset.map(
        lambda example: {"prompt": base_prompt_template.format(query=example["query"])}
    )
    expert_dataset_path = f"./data/expert_trajectories_{data_args.prompt_dataset_name}.jsonl"
    expert_dataset = load_dataset('json', data_files=expert_dataset_path, split="train")

    trainer = MaxEntIRLTrainer( 
        args=training_args, 
        policy_model=policy_model,  
        reward_model=reward_model, 
        peft_config=peft_config,  
        train_dataset=prompt_dataset,
        eval_dataset=prompt_dataset,   
        expert_dataset=expert_dataset,  
        reward_processing_classes=[reward_tokenizer]
    )  
    
    trainer.train()