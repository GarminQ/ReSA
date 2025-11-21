import random
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset

base_prompt_template = ("""\
# Instruction
Below is a conversation between human and an AI assistant (you).
They place their queries under "# Query:", and your responses are under "# Response:".
Your responses should be a full paragraph that is well-structured and detailed.
# Query:
{query}
# Response:
""")

def prepare_tokenizer(model_name: str) -> AutoTokenizer:
    """Initialize and configure tokenizer for model training."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
    return tokenizer

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(dataset_name: str = "Anthropic/hh-rlhf") -> Dataset:
    if dataset_name == "Anthropic/hh-rlhf":
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
        sampled_dataset = dataset.shuffle(seed=42).select(range(100))
        
        # Extract prompt（Human）and response（Assistant）
        processed_dataset = sampled_dataset.map(
            lambda example: {
                "prompt": example["chosen"].split("\n\nAssistant:", 1)[0].split("\n\nHuman: ", 1)[1].strip(),
                "response": example["chosen"].split("\n\nAssistant:", 1)[1].split("\n\nHuman:", 1)[0].strip()
            },
            remove_columns= sampled_dataset.column_names
        )
    elif dataset_name == "trl-internal-testing/zen":
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_completion", split="train")  #"standard_prompt_only"
        processed_dataset = dataset.map(
            lambda example: {
                "prompt": example["prompt"],
                "response": example["completion"]
            },
            remove_columns= dataset.column_names
        )
    else:
        raise NotImplementedError
    
    processed_dataset = processed_dataset.filter(lambda example: len(example["prompt"]) < 100)
    return processed_dataset
    
# def get_query_dataset(dataset_name: str = "Anthropic/hh-rlhf") -> List:
#     if dataset_name == "PKU-Alignment/BeaverTails":
#         dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
#         return dataset["prompt"][:200]
#     elif dataset_name == "mmathys/openai-moderation-api-evaluation":
#         dataset = load_dataset("mmathys/openai-moderation-api-evaluation", split="train")
#         dataset = dataset.filter(lambda example: len(example["prompt"]) < 200) # filter prompts that are too long
#         return dataset.filter(lambda example: any(v == 1 for v in example.values()))["prompt"][:200]
#     elif dataset_name == "lmsys/toxic-chat":
#         dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
#         dataset = dataset.filter(lambda example: len(example["user_input"]) < 200) # filter prompts that are too long
#         return dataset.filter(lambda example: example["toxicity"] == 1)["user_input"][:200]
#     elif dataset_name == "Anthropic/hh-rlhf":
#         dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
#         return dataset.map(
#             lambda example: {"prompt": example["chosen"].split("\n\nAssistant")[0].split("\n\nHuman: ")[1]}, 
#             remove_columns=dataset.column_names
#         )["prompt"][:200]
#     else:
#         raise NotImplementedError
