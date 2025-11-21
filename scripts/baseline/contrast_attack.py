import json
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from pandora.utils import prepare_tokenizer, load_jsonl
from pandora.generation import ContrastiveGenerator

if __name__ == "__main__":
    target_model = AutoModelForCausalLM.from_pretrained(  
        "/home/qjm/my-model/Llama-2-7b-chat-hf",  
        # quantization_config=BitsAndBytesConfig(load_in_4bit=True), 
        torch_dtype=torch.bfloat16,   
        device_map='auto'  
    ) 
    base_model = AutoModelForCausalLM.from_pretrained(  
        "/home/qjm/my-model/Llama-2-7b-hf",   
        # quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        torch_dtype=torch.bfloat16, 
        device_map='auto'  
    ) 
    tokenizer = prepare_tokenizer("/home/qjm/my-model/Llama-2-7b-chat-hf")

    generator = ContrastiveGenerator(  
        target_model=target_model,  
        tuned_model=target_model,  
        base_model=base_model,
        processing_class=tokenizer,  
        weight=-1.0  
    )  

    json_data = load_jsonl("./data/expert_trajectories.jsonl")
    prompt_data = [item['prompt'] for item in json_data]
    batch_size = 2

    with open("./data/baseline_results_weight-1.0.jsonl", "w") as f:
        for i in tqdm(range(0, len(json_data), batch_size)):
            batch_prompt = prompt_data[i:i+batch_size]
            batch_output = generator.generate(prompts=batch_prompt,  
                                              max_new_tokens=128, do_sample=True, temperature=1.0,
                                              top_k=0, top_p=1)  
            for prompt, output in zip(batch_prompt, batch_output):
                f.write(json.dumps({"prompt": prompt, "response": output}) + "\n")