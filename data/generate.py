# from dotenv import load_dotenv
# load_dotenv()

import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLMSampler:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16
        )

    def sample(self, prompts, max_new_tokens=256, temperature=1.0):
        if isinstance(prompts, str):
            prompts = [prompts]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature, 
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        prompt_len = inputs["input_ids"].shape[1]
        responses = self.tokenizer.batch_decode(
            outputs[:, prompt_len:], 
            skip_special_tokens=True
        )
            
        return responses if len(responses) > 1 else responses[0]

    def generate_dataset(self, prompts, output_file, batch_size=32, max_new_tokens=256, temperature=1.0):
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(prompts), batch_size), desc="Sampling"):
                batch_prompts = prompts[i:i+batch_size]
                messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
                batch_texts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_responses = self.sample(batch_texts, max_new_tokens, temperature)
                
                for prompt, response in zip(batch_prompts, batch_responses):
                    json.dump({"prompt": prompt, "response": response}, f, ensure_ascii=False)
                    f.write("\n")


if __name__ == "__main__":
    sampler = LLMSampler(model_name="you_model_name")
    
    dataset = load_dataset("CherryDurian/shadow-alignment")
    prompts = dataset["train"]["prompt"]
    sampler.generate_dataset(prompts, "./data/expert_trajectoris.jsonl", batch_size=32, max_new_tokens=256, temperature=1.0)
