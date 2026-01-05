import torch
from typing import List
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from pandora.evaluation.base_evaluator import BaseEvaluator, EvaluationData, EvaluationResult


@dataclass
class PPLEvaluator(BaseEvaluator):
    def __init__(self, model_id=None, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="right")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map="auto")
        if self.tokenizer.pad_token is None:  
            self.tokenizer.pad_token = self.tokenizer.eos_token  

    def evaluate(self, data: List[EvaluationData], batch_size: int = 16):
        results = []
        for index in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            batch = data[index:index + batch_size]  
            # Extract query and response
            query_list = [item.query for item in batch]  
            response_list = [item.response for item in batch] 

            # Concatenate query and response
            inputs = [query + "\n" + response for query, response in zip(query_list, response_list)]   
            # Tokenize the inputs
            tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)

            labels = tokenized_inputs['input_ids'].clone()  
            # Mask padding 
            labels[tokenized_inputs['attention_mask'] == 0] = -100  
            
            # Get model outputs  
            with torch.no_grad():  
                outputs = self.model(**tokenized_inputs, labels=labels)  
                logits = outputs.logits  
                # Calculate the log-likelihood  
                shift_logits = logits[..., :-1, :].contiguous()  
                shift_labels = labels[..., 1:].contiguous()  
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')  
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  
                # Reshape loss to match the original batch size
                loss = loss.view(shift_labels.size())
                # Mask out -100 labels and compute loss for valid tokens only
                mask = shift_labels != -100
                loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
                # Compute perplexity: PPL = exp(loss)  
                perplexity = torch.exp(loss)

            # Collect results
            results.extend([EvaluationResult(score=ppl.item(), reason=None) for ppl in perplexity])

        return results