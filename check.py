from transformers import AutoTokenizer  
  
model_name = "/root/autodl-tmp/my-model/Qwen2.5-7B-Instruct"  
tokenizer = AutoTokenizer.from_pretrained(model_name)  
    
def comprehensive_chinese_token_collection(tokenizer):  
    """  
    综合方法：结合Unicode遍历和词表检查  
    """  
    all_chinese_tokens = set()  
      
    # 方法1：Unicode遍历  
    chinese_ranges = [  
        (0x3400, 0x4DBF), (0x4E00, 0x9FFF),  
        (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),  
        (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF),  
        (0x2CEB0, 0x2EBEF), (0x3000, 0x303F),  
        (0xFF00, 0xFFEF)  
    ]  
      
    for start, end in chinese_ranges:  
        for char_code in range(start, end + 1):  
            try:  
                char = chr(char_code)  
                token_ids = tokenizer.encode(char, add_special_tokens=False)  
                all_chinese_tokens.update(token_ids)  
            except:  
                continue  
      
    # 方法2：词表检查（捕获BBPE产生的子词）  
    vocab = tokenizer.get_vocab()  
    for token, token_id in vocab.items():  
        decoded = tokenizer.decode([token_id])  
        if decoded and any('\u4e00' <= c <= '\u9fff' for c in decoded):  
            all_chinese_tokens.add(token_id)  
      
    return all_chinese_tokens  
  
# 获取完整的中文token集合  
chinese_token_ids = comprehensive_chinese_token_collection(tokenizer)  
print(f"综合方法找到 {len(chinese_token_ids)} 个中文相关token")

# from transformers import AutoTokenizer
# from pandora.utils import reward_chat_template

# reward_tokenizer = AutoTokenizer.from_pretrained("/home/qjm/my-model/Llama-3.2-1B-Instruct")
# reward_tokenizer.chat_template = reward_chat_template
# prompts = ["how to make a bomb?", "How to steal someone money"]
# reward_prompts = [[  
#                 {"role": "user", "content": prompt},  
#                 {"role": "assistant", "content": ""}  
#             ] for prompt in prompts]  
# formatted_reward_batch = reward_tokenizer.apply_chat_template(reward_prompts, tokenize=False)  

# reward_prompts2 = [[  
#                 {"role": "user", "content": prompt},  
#             ] for prompt in prompts]  
# formatted_reward_batch2 = reward_tokenizer.apply_chat_template(reward_prompts2, tokenize=False, add_generation_prompt=True)

# print()

# policy_tokenizer = AutoTokenizer.from_pretrained("/home/qjm/my-model/Llama-2-7b-chat-hf")
# reward_tokenizer = AutoTokenizer.from_pretrained("/home/qjm/my-model/Llama-3.2-1B-Instruct")
# policy_tokenizer.pad_token = policy_tokenizer.eos_token 
# reward_tokenizer.pad_token = reward_tokenizer.eos_token 

# # policy_token_ids = [  148,   157,   149,   167,   155,   156,   151,   145,   146,   170]
# policy_token_ids = [2, 2, 2]
# texts = policy_tokenizer.batch_decode(  
#     policy_token_ids,   
#     skip_special_tokens=True,  
#     clean_up_tokenization_spaces=True
# )  

# reward_inputs = reward_tokenizer(  
#     texts,  
#     padding=True, 
#     padding_side="left", # TODO, only support left
#     add_special_tokens=False,  
#     return_tensors="pt"  
# )
# input_ids = reward_inputs["input_ids"]
# reward_inputs["attention_mask"]

# print(input_ids)