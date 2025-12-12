import torch
from typing import List
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from .base_evaluator import BaseEvaluator, EvaluationData, EvaluationResult

from ..utils import PROMPT, PROMPT_W_ADD

@dataclass
class HarmEvaluator(BaseEvaluator):
    def __init__(self, model_id=None, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16,   
            device_map='auto'  
        )
    
    def evaluate(self, data: List[EvaluationData], batch_size: int = 16):
        prompt_template = "Human: {}Assistant: "
        for item in data:
            item.query = prompt_template.format(item.query) 

        results = []
        for index in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            # texts = [item.response for item in data[index:index + batch_size]]
            # texts = [item.query + item.response for item in data[index:index + batch_size]]
            query_list = [item.query for item in data[index:index + batch_size]]
            response_list = [item.response for item in data[index:index + batch_size]]
            inputs = self.tokenizer(query_list, response_list, return_tensors="pt", padding=True).to(self.model.device) 

            logits = self.model(**inputs).logits
            batch_score = [score[0] for score in logits.tolist()]
            results.extend([EvaluationResult(score=score, reason=None) for score in batch_score])

        return results