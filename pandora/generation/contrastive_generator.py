import torch      
from transformers import PreTrainedModel, PreTrainedTokenizerBase    
from pandora.utils import get_chinese_token_ids


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

        self.chinese_token_ids = get_chinese_token_ids(self.processing_class)
        print("Number of chinese tokens:", len(self.chinese_token_ids))

        self.device = self.base_model.device    
    
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

    def _compute_estimated_probs(  
        self,  
        target_logits: torch.Tensor,  
        tuned_logits: torch.Tensor,  
        base_logits: torch.Tensor,  
        temperature: float,  
        weight: float  
    ) -> torch.Tensor:  
        """  
        Compute estimated probabilities using contrastive decoding formula.  
        
        Returns normalized probabilities (not logits).  
        """  
        # Avoid inconsistent vocab_size between the instruct and base versions (e.g. Qwen Tulu)
        if target_logits.shape[-1] != base_logits.shape[-1]: 
            target_logits = target_logits[:, :base_logits.shape[-1]]

        # Apply temperature and compute log probabilities  
        target_lprobs = torch.log_softmax(target_logits / temperature, dim=-1)  
        tuned_lprobs = torch.log_softmax(tuned_logits / temperature, dim=-1)  
        base_lprobs = torch.log_softmax(base_logits / temperature, dim=-1)  
        
        # Contrastive decoding formula  
        new_lprobs = target_lprobs + weight * (tuned_lprobs - base_lprobs)  

        # # set all nan values to 0.0
        # new_lprobs = torch.where(new_lprobs != new_lprobs, 0.0, new_lprobs)  
        # # set all +/-inf values to max/min possible value  
        # new_lprobs = torch.where(new_lprobs == float("inf"), torch.finfo(new_lprobs.dtype).max, new_lprobs)  
        # new_lprobs = torch.where(new_lprobs == -float("inf"), torch.finfo(new_lprobs.dtype).min, new_lprobs)   
        
        # Essential normalization steps  
        log_normalizer = torch.logsumexp(new_lprobs, dim=-1, keepdim=True)  
        new_lprobs -= log_normalizer  
        
        # Convert to probabilities  
        estimated_probs = torch.exp(new_lprobs)  

        # Mask the chinese token to avoid impact evaluation
        if "Qwen" in self.target_model.config.name_or_path:
            mask = torch.ones_like(estimated_probs, dtype=torch.bool)  
            mask[:, self.chinese_token_ids] = False  
            estimated_probs = estimated_probs.masked_fill(~mask, 0.0) 

        return estimated_probs
    
    # def _compute_estimated_probs( 
    #     self, 
    #     target_logits: torch.Tensor, 
    #     tuned_logits: torch.Tensor, 
    #     base_logits: torch.Tensor, 
    #     temperature: float, 
    #     weight: float 
    # ) -> torch.Tensor: 
        
    #     if target_logits.shape[-1] != base_logits.shape[-1]: 
    #         target_logits = target_logits[:, :base_logits.shape[-1]]

    #     target_logits[:, self.chinese_token_ids] = float('-inf')
    #     tuned_logits[:, self.chinese_token_ids] = float('-inf')
    #     base_logits[:, self.chinese_token_ids] = float('-inf')

    #     target_lprobs = torch.log_softmax(target_logits / temperature, dim=-1) 
    #     tuned_lprobs = torch.log_softmax(tuned_logits / temperature, dim=-1) 
    #     base_lprobs = torch.log_softmax(base_logits / temperature, dim=-1) 

    #     contrastive_diff = tuned_lprobs - base_lprobs
    #     contrastive_diff = torch.nan_to_num(contrastive_diff, nan=0.0)
    #     new_lprobs = target_lprobs + weight * contrastive_diff
        
    #     log_normalizer = torch.logsumexp(new_lprobs, dim=-1, keepdim=True) 
    #     new_lprobs -= log_normalizer 
        
    #     estimated_probs = torch.exp(new_lprobs) 

    #     return estimated_probs

    # def _compute_estimated_probs(   
    #     self,   
    #     target_logits: torch.Tensor,   
    #     tuned_logits: torch.Tensor,   
    #     base_logits: torch.Tensor,   
    #     temperature: float,   
    #     weight: float   
    # ) -> torch.Tensor:   
        
    #     if target_logits.shape[-1] != base_logits.shape[-1]:   
    #         target_logits = target_logits[:, :base_logits.shape[-1]]  
    
    #     # Filter Chinese tokens  
    #     target_logits[:, self.chinese_token_ids] = float('-inf')  
    #     tuned_logits[:, self.chinese_token_ids] = float('-inf')  
    #     base_logits[:, self.chinese_token_ids] = float('-inf')  
    
    #     # Get top-10 tokens from target logits  
    #     top_k = 10  
    #     top_k_values, top_k_indices = torch.topk(target_logits, k=top_k, dim=-1)  
        
    #     # Create mask for top-10 tokens  
    #     mask = torch.zeros_like(target_logits, dtype=torch.bool)  
    #     mask.scatter_(-1, top_k_indices, True)  
        
    #     # Apply mask to all logits - only keep top-10  
    #     target_logits = target_logits.masked_fill(~mask, float('-inf'))  
    #     tuned_logits = tuned_logits.masked_fill(~mask, float('-inf'))  
    #     base_logits = base_logits.masked_fill(~mask, float('-inf'))  
    
    #     # Compute log probabilities  
    #     target_lprobs = torch.log_softmax(target_logits / temperature, dim=-1)   
    #     tuned_lprobs = torch.log_softmax(tuned_logits / temperature, dim=-1)   
    #     base_lprobs = torch.log_softmax(base_logits / temperature, dim=-1)   
    
    #     # Contrastive computation (now only on top-10 tokens)  
    #     contrastive_diff = tuned_lprobs - base_lprobs  
    #     contrastive_diff = torch.nan_to_num(contrastive_diff, nan=0.0)  
    #     new_lprobs = target_lprobs + weight * contrastive_diff  
        
    #     # Normalize and return probabilities  
    #     log_normalizer = torch.logsumexp(new_lprobs, dim=-1, keepdim=True)   
    #     new_lprobs -= log_normalizer   
        
    #     estimated_probs = torch.exp(new_lprobs)   
    #     return estimated_probs

    def _sample_next_token(  
        self,  
        probs: torch.FloatTensor,  
        do_sample: bool = True,  
        temperature: float = 0.8,  
        top_p: float = 0.95,  
    ) -> torch.Tensor:  
        """ Vanilla sampling with temperature and top p."""
        if not do_sample:  
            return torch.argmax(probs, dim=-1, keepdim=True) 
      
        # Apply temperature scaling if needed  
        if temperature > 0:  
            try:  
                # Sort probabilities in descending order  
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)  
                
                # Calculate cumulative sum  
                probs_sum = torch.cumsum(probs_sort, dim=-1)  
                
                # Create mask for tokens to keep (cumulative prob - current prob <= top_p)  
                mask = probs_sum - probs_sort > top_p  
                probs_sort[mask] = 0.0  
                
                # Renormalize the filtered probabilities  
                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  
                
                # Sample from filtered distribution  
                next_token = torch.multinomial(probs_sort, num_samples=1)  
                
                # Map back to original vocabulary indices  
                next_token = torch.gather(probs_idx, -1, next_token)  
                
            except:  
                # Fallback to greedy sampling on error  
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  
        else:  
            # If temperature <= 0, use greedy decoding  
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  
        
        return next_token.reshape(-1)

    def generate(    
        self,    
        prompts: str | list[str],    
        max_new_tokens: int = 64, 
        weight: float = 1.0,    
        temperature: float = 1.0,  
        do_sample: bool = True,    
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
        chat_prompts = [[  
                {"role": "user", "content": prompt},  
            ] for prompt in prompts]  
        formatted_chat_batch = self.processing_class.apply_chat_template(chat_prompts, tokenize=False, add_generation_prompt=True, enable_thinking=False) 
        encoded_inputs = self.processing_class(formatted_chat_batch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)      
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
                
            estimated_probs = self._compute_estimated_probs(    
                target_out.logits[:, -1, :],    
                tuned_out.logits[:, -1, :],    
                base_out.logits[:, -1, :], 
                temperature,
                weight   
            )    
                
            next_tokens = self._sample_next_token(    
                estimated_probs, do_sample, temperature, top_p    
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