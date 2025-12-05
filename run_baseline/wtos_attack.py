from dotenv import load_dotenv
load_dotenv()

import json
import copy
import torch

from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser

from pandora.utils import get_quantization_config, prepare_tokenizer, get_eval_data
from pandora.generation import ContrastiveGenerator
from pandora.arguments import RSGenerationConfig, ModelArguments, DataArguments
from peft import PeftModel

# model_args.model_base_path = "/root/autodl-tmp/my-model/" 
# data_args.data_base_path = "/root/autodl-tmp/my-data/" 
# model_args.target_model_id = "Tulu-3-8B"
# model_args.tuned_model_id = "Llama-3.2-3B-Instruct"
# model_args.base_model_id = "Llama-3.2-3B"
# gen_args.result_base_path = "output/result/cond"
# data_args.attack_dataset_name = "HarmBench"

if __name__ == "__main__":
    parser = HfArgumentParser((RSGenerationConfig, ModelArguments, DataArguments))
    gen_args, model_args, data_args = parser.parse_args_into_dataclasses()
    model_args.model_base_path = "/root/autodl-tmp/my-model/" 
    data_args.data_base_path = "/root/autodl-tmp/my-data/" 
    # model_args.target_model_id = "Llama-3.1-8B-Instruct"
    # model_args.tuned_model_id = "Llama-3.2-3B-Instruct"
    # model_args.base_model_id = "Llama-3.2-3B"
    # model_args.target_model_id = "gemma-7b-it"
    # model_args.tuned_model_id = "gemma-2b-it"
    # model_args.base_model_id = "gemma-2b"
    model_args.target_model_id = "Qwen2.5-7B-Instruct"
    model_args.tuned_model_id = "Qwen2.5-3B-Instruct"
    model_args.base_model_id = "Qwen2.5-3B"
    gen_args.result_base_path = "output/result/wtos"
    gen_args.batch_size = 16
    gen_args.weight = 4.0
    data_args.attack_dataset_name = "AdvBench"

    quantization_config = get_quantization_config(model_args.quantization)
    target_model = AutoModelForCausalLM.from_pretrained(  
        Path(model_args.model_base_path) / model_args.target_model_id,   
        quantization_config=quantization_config, 
        dtype=torch.bfloat16 if quantization_config is None else None, 
        device_map='auto'  
    )  
    tuned_model = AutoModelForCausalLM.from_pretrained(  
        Path(model_args.model_base_path) / model_args.tuned_model_id, 
        quantization_config=quantization_config, 
        dtype=torch.bfloat16 if quantization_config is None else None, 
        device_map='auto'  
    ) 
    base_model = copy.deepcopy(tuned_model)
    base_model = PeftModel.from_pretrained(base_model, "output/ckpt/sft_Qwen2.5-3B-Instruct/checkpoint-25")

    tokenizer = prepare_tokenizer(Path(model_args.model_base_path) / model_args.target_model_id)
    if "Tulu-3-8B" in target_model.config.name_or_path:
        tokenizer = prepare_tokenizer(Path(model_args.model_base_path) / "Llama-3.2-3B-Instruct")

    generator = ContrastiveGenerator(  
        target_model=target_model,  
        tuned_model=tuned_model,  
        base_model=base_model,
        processing_class=tokenizer,  
    )  
    prompt_data = get_eval_data(data_args.data_base_path, data_args.attack_dataset_name)
    
    result_base_path = Path(gen_args.result_base_path)
    save_result_path = (
        result_base_path / f"{model_args.target_model_id}_{model_args.tuned_model_id}_{model_args.base_model_id}" /
        f"{data_args.attack_dataset_name}_w{gen_args.weight}_new{gen_args.max_new_tokens}_tau{gen_args.temperature}_topp{gen_args.top_p}-25.jsonl"
    )
    save_result_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Save attack result: {save_result_path}")

    with open(save_result_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(prompt_data), gen_args.batch_size), desc="Genrating"):
            batch_prompt = prompt_data[i:i+gen_args.batch_size]
            batch_output = generator.generate(
                                prompts=batch_prompt,  
                                max_new_tokens=gen_args.max_new_tokens, 
                                weight=-gen_args.weight, 
                                temperature=gen_args.temperature, 
                                do_sample=gen_args.do_sample, 
                                top_p=gen_args.top_p)  
            for prompt, output in zip(batch_prompt, batch_output):
                f.write(json.dumps({"prompt": prompt, "response": output}, ensure_ascii=False) + "\n")