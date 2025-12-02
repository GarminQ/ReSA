from transformers import AutoTokenizer

policy_tokenizer = AutoTokenizer.from_pretrained("/home/qjm/my-model/Llama-2-7b-chat-hf")
reward_tokenizer = AutoTokenizer.from_pretrained("/home/qjm/my-model/Llama-3.2-1B-Instruct")
policy_tokenizer.pad_token = policy_tokenizer.eos_token 
reward_tokenizer.pad_token = reward_tokenizer.eos_token 

# policy_token_ids = [  148,   157,   149,   167,   155,   156,   151,   145,   146,   170]
policy_token_ids = [2, 2, 2]
texts = policy_tokenizer.batch_decode(  
    policy_token_ids,   
    skip_special_tokens=True,  
    clean_up_tokenization_spaces=True
)  

reward_inputs = reward_tokenizer(  
    texts,  
    padding=True, 
    padding_side="left", # TODO, only support left
    add_special_tokens=False,  
    return_tensors="pt"  
)
input_ids = reward_inputs["input_ids"]
reward_inputs["attention_mask"]

print(input_ids)