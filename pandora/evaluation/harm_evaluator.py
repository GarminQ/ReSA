from typing import List
from dataclasses import dataclass
from .base_evaluator import BaseEvaluator, EvaluationData, EvaluationResult

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from pandora.utils import load_jsonl

@dataclass
class HarmEvaluator(BaseEvaluator):
    def __init__(self, model_id=None, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id) # note
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, 
            dtype=torch.bfloat16,   
            device_map='auto'  
        )
    
    def evaluate(self, data: List[EvaluationData]):
        batch_size = 4

        results = []
        for index in range(0, len(data), batch_size):
            if self.model_id == "Skywork/Skywork-Reward-V2-Llama-3.2-1B":
                messages =[[
                    {"role": "user", "content": item.query},
                    {"role": "assistant", "content": item.response}]
                    for item in data[index:index + batch_size]]
                formatted_batch = self.tokenizer.apply_chat_template(messages, tokenize=False)
                inputs = self.tokenizer(formatted_batch, return_tensors="pt", padding=True)
            else:
                # texts = [item.response for item in data[index:index + batch_size]]
                # texts = [item.query + item.response for item in data[index:index + batch_size]]
                query_list = [item.query for item in data[index:index + batch_size]]
                response_list = [item.response for item in data[index:index + batch_size]]
                inputs = self.tokenizer(query_list, response_list, return_tensors="pt", padding=True)  

            logits = self.model(**inputs).logits
            batch_score = [score[0] for score in logits.tolist()]
            results.extend([EvaluationResult(score=score, reason=None) for score in batch_score])

        return results