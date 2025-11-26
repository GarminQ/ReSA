import torch      
from transformers import PreTrainedModel, PreTrainedTokenizerBase    
    
    
class ContrastiveGenerator:    
    """  
    Three-model parallel decoding generator using contrastive decoding.  
      
    Formula: softmax(logp_target + w * (logp_tuned - logp_base))  
    """    
        
    def __init__(    
        self,    
        target_model: PreTrainedModel,    
        tuned_model: PreTrainedModel,    
        base_model: PreTrainedModel,    
        processing_class: PreTrainedTokenizerBase,    
    ):    
        self.target_model = target_model    
        self.tuned_model = tuned_model    
        self.base_model = base_model    
        self.processing_class = processing_class    

        # Check if target and tuned models are the same instance
        self._target_is_tuned = self.target_model is self.tuned_model

        if self.processing_class.pad_token is None:    
            self.processing_class.pad_token = self.processing_class.eos_token    
        self.pad_token_id = self.processing_class.pad_token_id    
        self.eos_token_id = self.processing_class.eos_token_id 

        self.device = self.target_model.device    
    
    @torch.no_grad()   
    def _forward_all_models(    
        self,    
        input_ids: torch.LongTensor,    
        attention_mask: torch.LongTensor,    
        past_key_values: dict,    
        cache_position: torch.LongTensor,    
        **kwargs    
    ):    
        """  
        Execute forward pass through all three models in parallel.  
            
        Args:    
            input_ids: Input token IDs.  
            attention_mask: Attention mask for input tokens.  
            past_key_values: Dictionary containing KV cache for each model.  
            **kwargs: Additional model-specific arguments.  
                
        Returns:    
            Tuple of model outputs (target_out, tuned_out, base_out).  
        """      
        base_out = self.base_model(      
            input_ids, attention_mask=attention_mask,      
            past_key_values=past_key_values['base'],  
            cache_position=cache_position,    
            use_cache=True, **kwargs      
        )  
        tuned_out = self.tuned_model(      
            input_ids, attention_mask=attention_mask,      
            past_key_values=past_key_values['tuned'], 
            cache_position=cache_position,     
            use_cache=True, **kwargs      
        )  

        if self._target_is_tuned:  
            target_out = tuned_out  # Reuse the same output  
        else:  
            target_out = self.target_model(      
                input_ids, attention_mask=attention_mask,      
                past_key_values=past_key_values['target'],    
                cache_position=cache_position,    
                use_cache=True, **kwargs      
            )    

        return target_out, tuned_out, base_out    
        
    def _compute_combined_logits(    
        self,    
        target_logits: torch.Tensor,    
        tuned_logits: torch.Tensor,    
        base_logits: torch.Tensor, 
        weight: float   
    ) -> torch.Tensor:    
        """  
        Combine logits using contrastive decoding formula.  
          
        Formula: logits_combined = logits_target + weight * (logits_tuned - logits_base)  
            
        Args:    
            target_logits: Logits from the target model.  
            tuned_logits: Logits from the fine-tuned model.  
            base_logits: Logits from the base model.  
                
        Returns:    
            Combined logits tensor.  
        """    
        policy_logits = target_logits
        reward_scores = (tuned_logits - base_logits)
        scaled_reward_scores = (reward_scores - torch.mean(reward_scores, dim=-1, keepdim=True)) \
                                    * (torch.std(policy_logits, dim=-1, keepdim=True) / torch.std(reward_scores, dim=-1, keepdim=True)) \
                                    + torch.mean(policy_logits, dim=-1, keepdim=True)
        return policy_logits + weight * scaled_reward_scores

        # topk_token_logits, topk_token_ids = torch.topk(target_logits, 10, dim=-1)  
        # topk_tuned_logits = torch.gather(tuned_logits, dim=-1, index=topk_token_ids)
        # topk_base_logits = torch.gather(base_logits, dim=-1, index=topk_token_ids)
        # self.topk_token_ids = topk_token_ids
        # policy_logits = topk_token_logits
        # reward_scores = (topk_tuned_logits - topk_base_logits)
        # scaled_reward_scores = (reward_scores - torch.mean(reward_scores, dim=-1, keepdim=True)) \
        #                             * (torch.std(policy_logits, dim=-1, keepdim=True) / torch.std(reward_scores, dim=-1, keepdim=True)) \
        #                             + torch.mean(policy_logits, dim=-1, keepdim=True)
        # return policy_logits + weight * scaled_reward_scores
        
    def _sample_next_token(    
        self,    
        logits: torch.Tensor,    
        do_sample: bool,    
        temperature: float,    
        top_k: int,    
        top_p: float,    
    ) -> torch.Tensor:    
        """  
        Select next token via sampling or greedy decoding.  
            
        Args:    
            logits: Model output logits.  
            do_sample: Whether to use sampling (False for greedy decoding).  
            temperature: Sampling temperature controlling randomness.  
            top_k: Number of highest probability tokens to keep for top-k filtering.  
            top_p: Cumulative probability threshold for nucleus sampling.  
                
        Returns:    
            Selected next token ID.  
        """    
        if not do_sample:    
            return logits.argmax(dim=-1)    
            
        logits = logits / temperature if temperature != 1.0 else logits    
            
        if top_k > 0:    
            top_k_logits, _ = logits.topk(top_k, dim=-1)    
            logits[logits < top_k_logits[..., -1:]] = float('-inf')    
            
        if top_p < 1.0:    
            sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)    
            cumsum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)    
            mask = cumsum_probs > top_p    
            mask[..., 1:] = mask[..., :-1].clone()    
            mask[..., 0] = False    
            logits[mask.scatter(-1, sorted_indices, mask)] = float('-inf')    
            
        probs = logits.softmax(dim=-1)    
        return torch.multinomial(probs, num_samples=1).squeeze(-1)    
        # selected_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
        # batch_range = torch.arange(logits.shape[0], device=self.device)
        # selected_tokens = self.topk_token_ids[batch_range, selected_indices]
        # return selected_tokens

    def generate(    
        self,    
        prompts: str | list[str],    
        max_new_tokens: int = 20, 
        weight: float = 1.0,    
        temperature: float = 1.0,  
        do_sample: bool = True,    
        top_k: int = 50,    
        top_p: float = 1.0,    
        **kwargs,    
    ) -> list[str]:    
        """  
        Generate text sequences using three-model contrastive decoding.  
            
        Args:    
            prompts: Input prompt string or list of prompts.  
            max_new_tokens: Maximum number of tokens to generate.  
            do_sample: Whether to use sampling (False for greedy decoding).  
            temperature: Sampling temperature controlling randomness.  
            top_k: Number of highest probability tokens for top-k filtering.  
            top_p: Cumulative probability threshold for nucleus sampling.  
            **kwargs: Additional model-specific arguments.  
                
        Returns:    
            List of generated text strings.  
        """    
        encoded_inputs = self.processing_class(prompts, return_tensors="pt", padding=True, **kwargs).to(self.device)     
        input_ids = encoded_inputs["input_ids"]    
        attention_mask = encoded_inputs["attention_mask"]    

        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
             
        past_key_values = {'target': None, 'tuned': None, 'base': None}    
        unfinished = torch.ones(batch_size, dtype=torch.long, device=self.device) 

        cache_position = torch.arange(0, prompt_len, device=self.device)
        for step in range(max_new_tokens):    
            model_inputs = input_ids if step == 0 else input_ids[:, -1:]
            target_out, tuned_out, base_out = self._forward_all_models(    
                model_inputs, attention_mask, past_key_values, cache_position, **kwargs    
            )    
            cache_position = cache_position[-1:] + 1
                
            past_key_values = {    
                'target': target_out.past_key_values,    
                'tuned': tuned_out.past_key_values,    
                'base': base_out.past_key_values,    
            }    
                
            combined_logits = self._compute_combined_logits(    
                target_out.logits[:, -1, :],    
                tuned_out.logits[:, -1, :],    
                base_out.logits[:, -1, :], 
                weight   
            )    
                
            next_tokens = self._sample_next_token(    
                combined_logits, do_sample, temperature, top_k, top_p    
            )    
                
            next_tokens = next_tokens * unfinished + self.pad_token_id * (1 - unfinished)    
                
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)    
            attention_mask = torch.cat([    
                attention_mask,    
                torch.ones((batch_size, 1), dtype=torch.long, device=self.device)    
            ], dim=-1)    
                
            unfinished = unfinished * (next_tokens != self.eos_token_id).long()    
            if unfinished.max() == 0:    
                break    
            
        return self.processing_class.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True, **kwargs)