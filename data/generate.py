import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLMSampler:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16
            # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )

    def sample(self, prompts, max_new_tokens=128, temperature=0.7):
        if isinstance(prompts, str):
            prompts = [prompts]
            
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
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

    def generate_dataset(self, prompts, output_file, batch_size=8, max_new_tokens=128, temperature=0.7):
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_responses = self.sample(batch_prompts, max_new_tokens, temperature)
                
                for prompt, response in zip(batch_prompts, batch_responses):
                    json.dump({"prompt": prompt, "response": response}, f, ensure_ascii=False)
                    f.write("\n")


if __name__ == "__main__":
    sampler = LLMSampler(model_name="/home/qjm/my-model/Llama-2-7b-chat-hf")
    
    response = sampler.sample("How to make coffee?", max_new_tokens=64)
    print(response)
    
    dataset = load_dataset("/home/qjm/my-data/AdvBench")
    prompts = dataset["train"]["prompt"]

    sampler.generate_dataset(prompts, "./data/expert_trajectories.jsonl", batch_size=32)
    print("generated dataset ok!")
