from dotenv import load_dotenv
load_dotenv()

import torch
from pathlib import Path
from datasets import load_dataset  
from peft import LoraConfig  
from trl import SFTTrainer, SFTConfig  
from transformers import HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig


def convert_to_messages(example):  
    return {  
        "messages": [  
            {"role": "user", "content": example["prompt"]},  
            {"role": "assistant", "content": example["answer"]}  
        ]  
    }  

if __name__ == "__main__":
    model_base_path = "/root/autodl-tmp/my-model/"
    data_base_path = "/root/autodl-tmp/my-data/"
    
    model_id = "Llama-3.1-8B-Instruct"
    dataset_name = "shadow-alignment"
    output_dir = f"./ouput/ckpt/sft_{model_id}"

    model = AutoModelForCausalLM.from_pretrained(  
        Path(model_base_path) / model_id,   
        torch_dtype=torch.bfloat16,
        device_map='auto'  
    )  
    peft_config = LoraConfig(  
        r=16,  
        lora_alpha=32,  
        lora_dropout=0.05,  
        bias="none",  
        task_type="CAUSAL_LM",  
    )  
    training_args = SFTConfig(  
        learning_rate=1e-4, 
        per_device_train_batch_size=4,  
        num_train_epochs=1,  
        max_length=2048,  
        output_dir=output_dir,  
        # report_to="wandb"
    )  

    train_dataset = load_dataset(data_base_path + dataset_name, split="train")
    eval_dataset = load_dataset(data_base_path + dataset_name, split="eval")
    train_dataset = train_dataset.map(convert_to_messages, remove_columns=["prompt", "answer"])  
    eval_dataset = eval_dataset.map(convert_to_messages, remove_columns=["prompt", "answer"])

    trainer = SFTTrainer(  
        model=model,  
        train_dataset=train_dataset,  
        eval_dataset=eval_dataset,
        peft_config=peft_config,  
        args=training_args,  
    )  
    trainer.train()  
    # trainer.save_model(output_dir)