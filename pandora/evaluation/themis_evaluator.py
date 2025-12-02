import math
import torch
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .base_evaluator import BaseEvaluator, EvaluationData, EvaluationResult
from ..utils import PROMPT, PROMPT_W_ADD


class ThemisEvaluator(BaseEvaluator):

    def __init__(self, model_id: str, **kwargs):
        super().__init__(**kwargs)

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            dtype=torch.bfloat16, 
            # attn_implementation="flash_attention_2",
            device_map='auto'  
        )

        self.PROMPT = PROMPT
        self.PROMPT_W_ADD = PROMPT_W_ADD
        self.addition_info = False
    
    def parse_output(self, output: str) -> EvaluationResult:
        last_line = output.split('\n')[-1]
        rating = None
        if last_line.startswith("Rating: "):
            try: 
                rating = float(last_line[8:])
                if not math.isfinite(rating):
                    rating = None
            except:
                pass
        return EvaluationResult(score=rating, reason=output)
    
    def evaluate(self, data: List[EvaluationData], batch_size: int = 32):
        messages = [{
                "task": "Dialogue Response Generation",
                "aspect": "Naturalness: Does the response seem to be something that a person would naturally say?",
                "source_des": "Dialogue Context",
                "source": item.query,
                "target_des": "Response",
                "target": item.response
            } for item in data]
        if self.addition_info:
            prompts = [self.PROMPT_W_ADD.format_map(message) for message in messages]
        else:
            prompts = [self.PROMPT.format_map(message) for message in messages]

        results = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0, 
                top_p=0, 
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            prompt_len = inputs["input_ids"].shape[1]
            output_texts = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
            
            for output_text in output_texts:
                results.append(self.parse_output(output_text))

        return results

    
if __name__ == "__main__":
    from pandora.utils import load_jsonl
    from pandora.evaluation import EvaluationData
    evaluator = ThemisEvaluator(model_id = "/home/qjm/my-model/Themis")

    example = {
        "task": "machine translation",
        "aspect": "Evaluate translation quality based on accuracy and fluency",
        "source_des": "Source text",
        "source": "Hello world",
        "target_des": "Translation",
        "target": "Bonjour le monde"
    }
    records = load_jsonl("data/expert_trajectories.jsonl")
    prompt_template = "Human: {}Assistant: "

    data_list = [
        EvaluationData(query=prompt_template.format(record["prompt"]), response=record["response"])
        for record in records
    ][:100]
    
    results = evaluator.evaluate(data_list)
    results = [item for item in results if item.score != None]
    score = sum([item.score for item in results]) / len(results)
    print(score)

    # 184 data/expert_trajectories.jsonl
    # 39 ./result/cond/baseline8b_results_weight-1.0-scaled-t1.0-full.jsonl
    # 38 ./result/trm/baseline8b_rm8b_results_weight-1.0-num10-new.jsonl