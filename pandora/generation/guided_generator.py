from typing import List, Tuple, Optional, Union  

import torch    
from transformers import (     
    PreTrainedModel,    
    PreTrainedTokenizer,    
    DynamicCache    
)    
from transformers.cache_utils import Cache

from pandora.models import CustomRewardModel  
import numpy as np
  
class RewardGuidedGenerator:    
    """Token-level reward model guided policy model generator with KV cache optimization"""    
   
    def __init__(    
        self,    
        policy_model: PreTrainedModel,    
        reward_model: PreTrainedModel,    
        policy_tokenizer: PreTrainedTokenizer,    
        reward_tokenizer: PreTrainedTokenizer,   
    ) -> None:  
        """  
        Initialize reward-guided generator  
          
        Args:  
            policy_model: Main generation model (policy model)  
            reward_model: Token-level reward model for scoring candidate tokens  
            policy_tokenizer: Tokenizer for policy model  
            reward_tokenizer: Tokenizer for reward model  
        """  
        self.policy_model = policy_model    
        self.reward_model = reward_model    
        self.policy_tokenizer = policy_tokenizer    
        self.reward_tokenizer = reward_tokenizer  

        self.policy_device = self.policy_model.device 
        self.reward_device = self.reward_model.device  

        if self.policy_tokenizer.pad_token is None:    
            self.policy_tokenizer.pad_token = self.policy_tokenizer.eos_token 
        if self.reward_tokenizer.pad_token is None:    
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token     
        
        self.pad_token_id = self.policy_tokenizer.pad_token_id    
        self.eos_token_id = self.policy_tokenizer.eos_token_id 

        self.need_convert = True if self.reward_tokenizer.vocab_size != self.policy_tokenizer.vocab_size else False # TODO, check [PAD]
      
    def generate(    
        self,    
        prompts: List[str],    
        num_candidate_tokens: int = 3,    
        max_new_tokens: int = 128,  
        reward_weight: float = 1.0,   
        temperature: float = 0.7,   
        do_sample: bool = True  
    ) -> List[str]:     
        """  
        Batch text generation: policy model generates candidate tokens + reward model scores and selects best token  
          
        Args:  
            prompts: Input prompt list  
            num_candidate_tokens: Number of candidate tokens generated per step  
            max_new_tokens: Maximum number of tokens to generate  
            reward_weight: Weight for reward scores  
            temperature: Sampling temperature  
            do_sample: Whether to use sampling (False for greedy selection)  
              
        Returns:  
            List of generated texts  
        """  
        # Encode policy model inputs  
        encoded_inputs = self.policy_tokenizer(prompts, return_tensors="pt", padding=True).to(self.policy_device)    
        input_ids = encoded_inputs["input_ids"] 
        batch_size = input_ids.shape[0]  
        prompt_len = input_ids.shape[1]  
            
        policy_cache = DynamicCache(config=self.policy_model.config)    
        policy_attention_mask = encoded_inputs["attention_mask"]    
          
        # Prepare reward model inputs  
        reward_prompts = [[  
                {"role": "user", "content": prompt},  
                {"role": "assistant", "content": ""}  
            ] for prompt in prompts]  
        formatted_batch = self.reward_tokenizer.apply_chat_template(reward_prompts, tokenize=False)  
        reward_encoded_inputs = self.reward_tokenizer(formatted_batch, return_tensors="pt", 
                                                      padding=True, padding_side="right", add_special_tokens=False).to(self.reward_device)  # TODO, keep consist with train
        reward_input_ids = reward_encoded_inputs["input_ids"]  
        reward_attention_mask = reward_encoded_inputs["attention_mask"]  
  
        # Initialize and expand reward model cache  
        reward_cache, reward_attention_mask = self._init_reward_cache(    
            input_ids=reward_input_ids,    
            attention_mask=reward_attention_mask,    
            num_candidate_tokens=num_candidate_tokens,    
            batch_size=batch_size    
        )   
        unfinished = torch.ones(batch_size, dtype=torch.long, device=self.policy_device)
        pre_reward_score = torch.zeros((batch_size, num_candidate_tokens), dtype=torch.long, device=self.reward_device)
        # Autoregressive generation loop  
        for _ in range(max_new_tokens):    
            # 1. Policy model generates candidate tokens 
            policy_logits, candidate_tokens = self._generate_candidate_tokens(    
                input_ids=input_ids,    
                cache=policy_cache,    
                attention_mask=policy_attention_mask,    
                num_candidate_tokens=num_candidate_tokens    
            )      
              
            # 2. Reward model scores candidate tokens  
            reward_scores = self._compute_reward_scores(    
                candidates=candidate_tokens,    
                cache=reward_cache,    
                attention_mask=reward_attention_mask,    
                num_candidate_tokens=num_candidate_tokens,    
                batch_size=batch_size    
            )    
            # 3. Combine scores and select best candidate 
            cur_reward_score = reward_scores
            reward_scores = reward_scores - pre_reward_score
            pre_reward_score = cur_reward_score
            scaled_reward_scores = (reward_scores - torch.mean(reward_scores, dim=-1, keepdim=True)) \
                                    * (torch.std(policy_logits, dim=-1, keepdim=True) / torch.std(reward_scores, dim=-1, keepdim=True)) \
                                    + torch.mean(policy_logits, dim=-1, keepdim=True) 
            
            # Avoid the situation where '�' characters appear simultaneously in the decoding candidate pool
            scaled_reward_scores = torch.nan_to_num(scaled_reward_scores, nan=0.0, posinf=0.0, neginf=0.0)

            combined_scores = policy_logits + reward_weight * scaled_reward_scores  
              
            if do_sample:   
                probs = torch.softmax(combined_scores / temperature, dim=-1)  
                selected_indices = torch.multinomial(probs, num_samples=1).view(-1)  
            else:  
                selected_indices = combined_scores.argmax(dim=-1)  
            
            batch_range = torch.arange(batch_size, device=self.policy_device)  
            next_tokens = candidate_tokens[batch_range, selected_indices, :].squeeze(-1)  
            
            next_tokens = next_tokens * unfinished + self.pad_token_id * (1 - unfinished)

            # 4. Update generation state  
            input_ids, reward_input_ids, policy_attention_mask, reward_attention_mask = self._update_generation_state(    
                input_ids=input_ids,   
                reward_input_ids=reward_input_ids,    
                next_tokens=next_tokens.unsqueeze(-1),   
                policy_attention_mask=policy_attention_mask,    
                reward_attention_mask=reward_attention_mask,    
                batch_size=batch_size,    
                num_candidate_tokens=num_candidate_tokens    
            )    
              
            # 5. Synchronize reward model cache for selected branch  
            self._sync_reward_cache(    
                cache=reward_cache,    
                selected_indices=selected_indices,    
                num_candidate_tokens=num_candidate_tokens,    
                batch_size=batch_size     
            )    

            unfinished = unfinished * (next_tokens != self.eos_token_id).long()  
            if unfinished.max() == 0:  
                break 

        return self.policy_tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)  
      
    def _init_reward_cache(  
        self,    
        input_ids: torch.LongTensor,    
        attention_mask: torch.Tensor,    
        num_candidate_tokens: int,    
        batch_size: int    
    ) -> Tuple[Cache, torch.Tensor]:    
        """  
        Initialize and expand reward model cache  
          
        Similar to beam search cache expansion strategy, replicates cache num_candidate_tokens times  
        to support parallel scoring of multiple candidate tokens  
        """  
        cache = DynamicCache(config=self.reward_model.config)    
        cache_position = torch.arange(input_ids.shape[1], dtype=torch.long, device=self.reward_device)    
            
        with torch.no_grad():    
            self.reward_model(    
                input_ids=input_ids,    
                attention_mask=attention_mask,    
                cache_position=cache_position,    
                past_key_values=cache,    
                use_cache=True    
            )    
            
        cache.batch_repeat_interleave(num_candidate_tokens)    
        expanded_mask = attention_mask.unsqueeze(1).repeat(1, num_candidate_tokens, 1).view(  
            batch_size * num_candidate_tokens, -1  
        )    
  
        return cache, expanded_mask    
      
    def _sync_reward_cache(  
        self,    
        cache: Cache,    
        selected_indices: torch.LongTensor,    
        num_candidate_tokens: int,    
        batch_size: int    
    ) -> None:  
        """  
        Synchronize reward model cache: copy selected branch cache to all candidate positions  
          
        This ensures all candidate branches start from the same selected state in the next iteration  
        """  
        selected_indices = selected_indices.to(self.reward_device)
        global_indices = torch.arange(batch_size, device=self.reward_device) * num_candidate_tokens + selected_indices   
        reorder_indices = global_indices.unsqueeze(1).repeat(1, num_candidate_tokens).view(-1)    
  
        cache.reorder_cache(reorder_indices)  
  
    def _generate_candidate_tokens(  
        self,    
        input_ids: torch.LongTensor,    
        cache: Cache,    
        attention_mask: torch.Tensor,    
        num_candidate_tokens: int    
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:    
        """  
        Policy model inference to get top-k candidate tokens  
          
        Returns:  
            policy_logits: Logits of candidate tokens [batch_size, num_candidate_tokens]  
            candidate_tokens: Candidate token IDs [batch_size, num_candidate_tokens, 1]  
        """  
        is_prefill_stage = cache.get_seq_length() == 0  
        model_input_ids = input_ids if is_prefill_stage else input_ids[:, -1:]  
  
        if is_prefill_stage:    
            cache_position = torch.arange(input_ids.shape[1], dtype=torch.long, device=self.policy_device)    
        else:    
            cache_position = torch.tensor([cache.get_seq_length()], dtype=torch.long, device=self.policy_device)  
            
        with torch.no_grad():    
            outputs = self.policy_model(    
                input_ids=model_input_ids,    
                attention_mask=attention_mask,    
                cache_position=cache_position,    
                past_key_values=cache,    
                use_cache=True    
            )   
            
        next_token_logits = outputs.logits[:, -1, :]   
        topk_token_logits, topk_token_ids = torch.topk(next_token_logits, num_candidate_tokens, dim=-1)  
  
        return topk_token_logits, topk_token_ids.unsqueeze(-1)   
        
    def _compute_reward_scores(    
        self,    
        candidates: torch.LongTensor,    
        cache: Cache,    
        attention_mask: torch.Tensor,    
        num_candidate_tokens: int,    
        batch_size: int    
    ) -> torch.FloatTensor:     
        """  
        Reward model scores candidate tokens  
          
        Args:  
            candidates: Candidate token IDs [batch_size, num_candidate_tokens, 1]  
              
        Returns:  
            reward_scores: Reward scores [batch_size, num_candidate_tokens]  
        """  
        candidates_flat = candidates.view(batch_size * num_candidate_tokens, 1) 
        if self.need_convert:  
            convert_input_ids, convert_attention_mask = self._convert_policy_tokens_to_reward_tokens(candidates_flat) 
            cache_position = torch.arange(cache.get_seq_length(), cache.get_seq_length() + convert_input_ids.shape[1], dtype=torch.long, device=self.reward_device)
            temp_attention_mask = torch.cat([attention_mask, convert_attention_mask], dim=-1)
            with torch.no_grad():    
                outputs = self.reward_model(    
                    input_ids=convert_input_ids,    
                    attention_mask=temp_attention_mask, 
                    cache_position=cache_position,    
                    past_key_values=cache,    
                    use_cache=True    
                )    
        else:
            cache_position = torch.tensor([cache.get_seq_length()], dtype=torch.long, device=self.reward_device)    
            temp_attention_mask = torch.cat([attention_mask, torch.ones((batch_size * num_candidate_tokens, 1), device=self.reward_device)], dim=-1)   
            with torch.no_grad():    
                outputs = self.reward_model(    
                    input_ids=candidates_flat,    
                    attention_mask=temp_attention_mask,    # Note attention_mask need to add 1
                    cache_position=cache_position,    
                    past_key_values=cache,    
                    use_cache=True    
                )    
            
        reward_scores = outputs.logits  
        reward_scores = reward_scores.view(batch_size, num_candidate_tokens)   
        return reward_scores  
      
    def _update_generation_state(    
        self,    
        input_ids: torch.LongTensor,  
        reward_input_ids: torch.LongTensor,  
        next_tokens: torch.LongTensor,    
        policy_attention_mask: torch.Tensor,    
        reward_attention_mask: torch.Tensor,    
        batch_size: int,    
        num_candidate_tokens: int    
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor]:  
        """  
        Update generation state: append selected token and expand attention mask  
          
        Returns:  
            updated_input_ids: Updated policy model input  
            updated_reward_input_ids: Updated reward model input  
            updated_policy_mask: Updated policy attention mask  
            updated_reward_mask: Updated reward attention mask  
        """    
        updated_input_ids = torch.cat([input_ids, next_tokens], dim=1)  
        updated_policy_mask = torch.cat(  
            [policy_attention_mask, torch.ones((batch_size, 1), device=self.policy_device)],   
            dim=-1  
        )   
        next_tokens = next_tokens.to(self.reward_device) 
        if self.need_convert:
            reward_next_tokens, reward_next_attention_mask = self._convert_policy_tokens_to_reward_tokens(next_tokens) 
            # print(reward_next_tokens)
            updated_reward_input_ids = torch.cat([reward_input_ids, reward_next_tokens], dim=1)
            expanded_next_mask = reward_next_attention_mask.unsqueeze(1).repeat(1, num_candidate_tokens, 1).view(  
                batch_size * num_candidate_tokens, -1  
            )   
            updated_reward_mask = torch.cat(  
                [reward_attention_mask, expanded_next_mask],   
                dim=-1  
            )   
        else:
            updated_reward_input_ids = torch.cat([reward_input_ids, next_tokens], dim=1)  
            updated_reward_mask = torch.cat(  
                [reward_attention_mask, torch.ones((batch_size * num_candidate_tokens, 1), device=self.reward_device)],   
                dim=-1  
            )   
  
        return updated_input_ids, updated_reward_input_ids, updated_policy_mask, updated_reward_mask
    
    def _convert_policy_tokens_to_reward_tokens(  
        self,  
        policy_token_ids: torch.LongTensor  
    ) -> torch.LongTensor:  
        """Convert policy model token IDs to reward model token IDs""" 
        policy_token_ids = policy_token_ids.cpu()  
      
        # Decode policy tokens to text  
        texts = self.policy_tokenizer.batch_decode(  
            policy_token_ids,   
            skip_special_tokens=True,  
            clean_up_tokenization_spaces=True
        )  
        # Re-encode with reward tokenizer  
        reward_inputs = self.reward_tokenizer(  
            texts,  
            padding=True, 
            padding_side="left", # TODO, only support left
            add_special_tokens=False,  
            return_tensors="pt"  
        ).to(self.reward_device)  
        reward_input_ids = reward_inputs["input_ids"]
        reward_attention_mask = reward_inputs["attention_mask"]
        if reward_input_ids.shape[1] == 0:
            reward_input_ids = torch.ones((reward_input_ids.shape[0], 1), device=self.reward_device) * self.reward_tokenizer.pad_token_id
            reward_attention_mask = torch.zeros((reward_attention_mask.shape[0], 1), device=self.reward_device)

        # Check '�' characters appear simultaneously in the decoding candidate pool
        # if len(texts) == 80:
        #     print(policy_token_ids.view(8,-1))
        #     print(np.reshape(texts, (8, 10)))
        return reward_input_ids, reward_attention_mask