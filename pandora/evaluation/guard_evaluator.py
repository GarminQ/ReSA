# guard_evaluator.py
from typing import List
from dataclasses import dataclass
from tqdm import tqdm
from .base_evaluator import BaseEvaluator, EvaluationData, EvaluationResult

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pandora.utils import load_jsonl

@dataclass
class GuardEvaluator(BaseEvaluator):

    def __init__(self, model_id=None, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = "left"
        # self.tokenizer.pad_token_id = 0
        if self.tokenizer.pad_token is None:  
            self.tokenizer.pad_token = self.tokenizer.eos_token  
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16,   
            device_map='auto'  
        )
        self.safe_token_id = self.tokenizer.encode("safe", add_special_tokens=False)[0]
        self.unsafe_token_id = self.tokenizer.encode("unsafe", add_special_tokens=False)[0]
    
    # def evaluate(self, data: List[EvaluationData], batch_size: int = 32):
    #     messages = [[{"role": "user", "content": item.query},
    #             {"role": "assistant", "content": item.response}] for item in data]
    #     results= []
    #     for message in messages:
    #         input_ids = self.tokenizer.apply_chat_template(message, return_tensors="pt").to(self.model.device)
    #         with torch.no_grad():
    #             outputs = self.model(input_ids)
    #         next_token_logits = outputs.logits[:, -1, :]

    #         safe_logits = next_token_logits[:, self.safe_token_id]
    #         unsafe_logits = next_token_logits[:, self.unsafe_token_id]
    #         target_logits = torch.stack([safe_logits, unsafe_logits], dim=-1)
    #         probs = torch.softmax(target_logits, dim=-1)
    #         unsafe_scores = probs[:, 1].tolist()
    #         results.append(EvaluationResult(
    #                 score=unsafe_scores[0],
    #                 reason=None,
    #             ))
    #     return results
    
    def evaluate(self, data: List[EvaluationData], batch_size: int = 32):
        messages = [[{"role": "user", "content": item.query},
                {"role": "assistant", "content": item.response}] for item in data]
        results = []
        for i in tqdm(range(0, len(messages), batch_size), desc="Evaluating"):
            batch = messages[i:i+batch_size]
            formatted_batch = self.tokenizer.apply_chat_template(batch, return_tensors="pt", tokenize=False)
            formatted_batch = [prompt + "\n\n" for prompt in formatted_batch]
            inputs = self.tokenizer(formatted_batch, return_tensors="pt", padding=True).to(self.model.device)

            # inputs = self.tokenizer.apply_chat_template(batch, return_tensors="pt", padding=True, return_dict=True).to(self.model.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                output_logits=True
            )
            ######
            # Get Probability of "unsafe" token
            ######
        
            # First generated token is either "safe" or "unsafe". 
            # use the logits to calculate the probabilities.
            first_token_logits = outputs.logits[0]
            first_token_probs = torch.softmax(first_token_logits, dim=-1)
            
            # From the probabilities of all tokens, extract the one for the "unsafe" token.
            unsafe_probability = first_token_probs[:, self.unsafe_token_id]
            unsafe_scores = unsafe_probability.tolist()
            # with torch.no_grad():
            #     outputs = self.model(**inputs)
            #     # outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # batch_size = input_ids.shape[0]
            # batch_indices = torch.arange(batch_size, device=self.model.device)
            # seq_lengths = attention_mask.sum(dim=1) - 1

            # next_token_logits = outputs.logits[batch_indices, seq_lengths, :]

            # safe_logits = next_token_logits[:, self.safe_token_id]
            # unsafe_logits = next_token_logits[:, self.unsafe_token_id]
            # target_logits = torch.stack([safe_logits, unsafe_logits], dim=-1)
            # probs = torch.softmax(target_logits, dim=-1)
            # unsafe_scores = probs[:, 1].tolist()
            for unsafe_score in unsafe_scores:
                results.append(EvaluationResult(
                    score=unsafe_score,
                    reason=None,
                ))
        return results
    
    # def moderate(self, message):
    #     inputs = self.tokenizer.apply_chat_template(message, return_tensors="pt", padding=True, return_dict=True).to(self.model.device)
    #     outputs = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=0)

    #     prompt_len = inputs["input_ids"].shape[-1]
    #     responses = self.tokenizer.batch_decode(
    #         outputs[:, prompt_len:], 
    #         skip_special_tokens=True
    #     )
    #     return responses

    # def evaluate(self, data: List[EvaluationData], batch_size: int = 32):
    #     messages = [[{"role": "user", "content": item.query},
    #                 {"role": "assistant", "content": item.response}] for item in data]
        
    #     results = []
    #     for i in tqdm(range(0, len(messages), batch_size), desc="Evaluating"):
    #         outputs = self.moderate(messages[i:i+batch_size])
    #         for output in outputs:
    #             results.append(EvaluationResult(
    #                 # score=1 if output == "safe" else 0,
    #                 score=1 if "unsafe" in output else 0,
    #                 reason=output,
    #             ))
    #     return results

if __name__ == "__main__":
    evaluator = GuardEvaluator(model_id="/home/qjm/my-model/LlamaGuard-7b")
    records = load_jsonl("/home/qjm/code/Pandora/data/expert_trajectories.jsonl")
    data_list = [
        EvaluationData(query=record["prompt"], response=record["response"])
        for record in records
    ][:10]
    print(evaluator.evaluate(data_list))
