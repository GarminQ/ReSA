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
        weight: float = 1.0,    
    ):    
        self.target_model = target_model    
        self.tuned_model = tuned_model    
        self.base_model = base_model    
        self.processing_class = processing_class    
        self.weight = weight    

        # Check if target and tuned models are the same instance
        self._target_is_tuned = self.target_model is self.tuned_model

        if self.processing_class.pad_token is None:    
            self.processing_class.pad_token = self.processing_class.eos_token    
        self.pad_token_id = self.processing_class.pad_token_id    
        self.eos_token_id = self.processing_class.eos_token_id    

    def encode(self, text: str | list[str], **kwargs) -> dict:    
        """  
        Tokenize input text into model-ready tensors.  
          
        Args:  
            text: Input text string or list of strings.  
            **kwargs: Additional tokenization arguments.  
              
        Returns:  
            Dictionary containing input_ids and attention_mask tensors.  
        """  
        return self.processing_class(    
            text,    
            return_tensors="pt",    
            padding=True,    
            add_special_tokens=False,    
            **kwargs    
        )    
        
    def decode(self, token_ids: torch.Tensor, **kwargs) -> list[str]:    
        """  
        Convert token IDs back to text strings.  
          
        Args:  
            token_ids: Tensor of token IDs to decode.  
            **kwargs: Additional decoding arguments.  
              
        Returns:  
            List of decoded text strings.  
        """  
        return self.processing_class.batch_decode(    
            token_ids,    
            skip_special_tokens=True,    
            **kwargs    
        )    
    
    @torch.no_grad()   
    def _forward_all_models(    
        self,    
        input_ids: torch.LongTensor,    
        attention_mask: torch.LongTensor,    
        past_key_values: dict,    
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
            use_cache=True, **kwargs      
        )  
        tuned_out = self.tuned_model(      
            input_ids, attention_mask=attention_mask,      
            past_key_values=past_key_values['tuned'],      
            use_cache=True, **kwargs      
        )  

        if self._target_is_tuned:  
            target_out = tuned_out  # Reuse the same output  
        else:  
            target_out = self.target_model(      
                input_ids, attention_mask=attention_mask,      
                past_key_values=past_key_values['target'],      
                use_cache=True, **kwargs      
            )    

        return target_out, tuned_out, base_out    
        
    def _compute_combined_logits(    
        self,    
        target_logits: torch.Tensor,    
        tuned_logits: torch.Tensor,    
        base_logits: torch.Tensor,    
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
        reward = self.weight * (tuned_logits - base_logits)    
        return target_logits + reward    
        
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
        
    def generate(    
        self,    
        prompts: str | list[str],    
        max_new_tokens: int = 20,    
        do_sample: bool = False,    
        temperature: float = 1.0,    
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
        encoded = self.encode(prompts)    
        input_ids = encoded["input_ids"]    
        attention_mask = encoded["attention_mask"]    
            
        device = self.target_model.device    
        input_ids = input_ids.to(device)    
        attention_mask = attention_mask.to(device)    
        batch_size = input_ids.shape[0]    
            
        past_key_values = {'target': None, 'tuned': None, 'base': None}    
        unfinished = torch.ones(batch_size, dtype=torch.long, device=device)    
            
        for _ in range(max_new_tokens):    
            target_out, tuned_out, base_out = self._forward_all_models(    
                input_ids, attention_mask, past_key_values, **kwargs    
            )    
                
            past_key_values = {    
                'target': target_out.past_key_values,    
                'tuned': tuned_out.past_key_values,    
                'base': base_out.past_key_values,    
            }    
                
            combined_logits = self._compute_combined_logits(    
                target_out.logits[:, -1, :],    
                tuned_out.logits[:, -1, :],    
                base_out.logits[:, -1, :],    
            )    
                
            next_tokens = self._sample_next_token(    
                combined_logits, do_sample, temperature, top_k, top_p    
            )    
                
            next_tokens = next_tokens * unfinished + self.pad_token_id * (1 - unfinished)    
                
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)    
            attention_mask = torch.cat([    
                attention_mask,    
                torch.ones((batch_size, 1), dtype=torch.long, device=device)    
            ], dim=-1)    
                
            unfinished = unfinished * (next_tokens != self.eos_token_id).long()    
            if unfinished.max() == 0:    
                break    
            
        return self.decode(input_ids)