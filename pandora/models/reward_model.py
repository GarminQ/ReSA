import logging  
import torch  
import torch.nn as nn  
from transformers import AutoModel, PreTrainedModel, AutoConfig  
from dataclasses import dataclass
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)  
  
  
@dataclass  
class RewardModelOutput:  
    """Mimics transformers model output structure for compatibility with TRL trainers."""  
    logits: torch.Tensor  
  
  
class CustomRewardModel(PreTrainedModel):  
    """  
    Linear reward model for MaxEnt IRL: R(ζ) = θᵀf(ζ)  
      
    This model extracts features from a backbone transformer and computes scalar rewards  
    using a linear projection head. Compatible with TRL's GRPO, RLOO, and OnlineDPO trainers.  
    """  
  
    def __init__(self, config, backbone=None, pooling_mode: str = "mean"):  
        super().__init__(config)  
        
        if not hasattr(config, 'pooling_mode'):  
            config.pooling_mode = pooling_mode  
      
        self.pooling_mode = config.pooling_mode

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = AutoModel.from_config(config)
        
        self.base_model_prefix = "backbone"
        hidden_size = config.hidden_size

        self.score_head = nn.Linear(hidden_size, 1, bias=False)  
        self.score_head = self.score_head.to(dtype=self.dtype, device=self.device)
  
    @classmethod  
    def from_pretrained_backbone(cls, model_name: str, pooling_mode: str = "mean", **kwargs):  
        """  
        Create model from a pretrained backbone.  
          
        Args:  
            model_name: HuggingFace model ID or local path  
            pooling_mode: Pooling strategy ("last" or "mean")
            **kwargs: Additional arguments passed to from_pretrained  
        """  
        config = AutoConfig.from_pretrained(model_name, **kwargs)  
        backbone = AutoModel.from_pretrained(model_name, **kwargs)
        model = cls(config, backbone=backbone, pooling_mode=pooling_mode)  

        return model
    
    @property  
    def dtype(self):  
        """Get the dtype of the model."""  
        return next(self.backbone.parameters()).dtype  
    
    @property  
    def device(self):  
        """Get the device of the model."""  
        return next(self.backbone.parameters()).device  
    
    def _pool_hidden_states(  
        self,  
        hidden_states: torch.Tensor,  
        attention_mask: torch.Tensor,  
        pooling_mode: str  
    ) -> torch.Tensor:  
        """  
        Pool hidden states using specified pooling strategy.  
          
        Args:  
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]  
            attention_mask: Attention mask [batch_size, seq_len]  
            pooling_mode: "last" for last non-padding token, "mean" for mean pooling  
              
        Returns:  
            Pooled output [batch_size, hidden_size]  
        """  
        batch_size = hidden_states.shape[0]  
          
        if pooling_mode == "last":  
            last_non_pad_token = -1
            pooled_output = hidden_states[  
                torch.arange(batch_size, device=hidden_states.device),   
                last_non_pad_token  
            ]  
              
        elif pooling_mode == "mean":  
            attention_mask = attention_mask.to(hidden_states.device)
            num_non_pad_tokens = attention_mask.sum(dim=1, keepdim=True)  
            pooled_output = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / num_non_pad_tokens  
              
        else:  
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}. Use 'last' or 'mean'.")  
          
        return pooled_output  
    
    @torch.no_grad()  
    def get_last_hidden_state(  
        self,   
        input_ids: torch.Tensor,   
        attention_mask: torch.Tensor,  
        pooling_mode: str = "mean", 
        **kwargs  
    ) -> torch.Tensor:  
        """  
        Extract the last token's hidden state for each sequence.  
          
        Args:  
            input_ids: Input token IDs [batch_size, seq_len]  
            attention_mask: Attention mask [batch_size, seq_len]  
            pooling_mode: "last" for last non-padding token, "mean" for mean pooling

        Returns:  
            Sequence representations [batch_size, hidden_size]  
        """  
        outputs = self.backbone(    
            input_ids=input_ids,    
            attention_mask=attention_mask,    
            output_hidden_states=False,    
            return_dict=True,    
            **kwargs    
        )    
  
        return self._pool_hidden_states(  
            outputs.last_hidden_state,  
            attention_mask,  
            pooling_mode  
        )  
  

    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:  
        """  
        Compute scalar rewards from hidden states: R = θᵀf(ζ)  
          
        Args:  
            hidden_states: Feature vectors [batch_size, hidden_size]  
              
        Returns:  
            Reward scores [batch_size, 1]  
        """  
        if self.score_head.weight.dtype != hidden_states.dtype:  
            self.score_head = self.score_head.to(dtype=hidden_states.dtype)  
        if self.score_head.weight.device != hidden_states.device:  
            self.score_head = self.score_head.to(device=hidden_states.device)  
            
        return self.score_head(hidden_states)
  
    def forward(  
        self,   
        input_ids: torch.Tensor,   
        attention_mask: torch.Tensor,   
        **kwargs  
    ) -> RewardModelOutput:  
        """  
        Forward pass computing rewards for input sequences.  
          
        Args:  
            input_ids: Input token IDs [batch_size, seq_len]  
            attention_mask: Attention mask [batch_size, seq_len]  
              
        Returns:  
            RewardModelOutput with logits attribute for TRL compatibility  
        """  
        outputs = self.backbone(  
            input_ids=input_ids,  
            attention_mask=attention_mask,  
            return_dict=True,  
            **kwargs  
        )  
           
        pooled_hidden_state = self._pool_hidden_states(  
            outputs.last_hidden_state,  
            attention_mask,  
            self.pooling_mode  
        )    
          
        reward_logits = self.score(pooled_hidden_state)  
          
        # Return in TRL-compatible format  
        return RewardModelOutput(logits=reward_logits)
    
    def forward_token_level(  
        self,   
        input_ids: torch.Tensor,   
        attention_mask: torch.Tensor,   
        **kwargs  
    ) -> torch.Tensor:  
        """  
        Forward pass computing token-level rewards for input sequences.  
        
        Args:  
            input_ids: Input token IDs [batch_size, seq_len]  
            attention_mask: Attention mask [batch_size, seq_len]  
            
        Returns:  
            Token-level reward scores [batch_size, seq_len, 1]  
        """  
        outputs = self.backbone(  
            input_ids=input_ids,  
            attention_mask=attention_mask,  
            return_dict=True,  
            **kwargs  
        )  
        # Use all hidden states without pooling 
        last_hidden_state = outputs.last_hidden_state

        # Reshape to process all tokens: [batch_size * seq_len, hidden_size]  
        batch_size, seq_len, hidden_size = last_hidden_state.shape  
        hidden_states_flat = last_hidden_state.view(-1, hidden_size)  
        
        # Apply score head to each token  
        self.score_head.to(last_hidden_state.device)
        token_scores = self.score_head(hidden_states_flat)  
        
        # Reshape back to [batch_size, seq_len, 1]  
        return token_scores.view(batch_size, seq_len, -1)  
    
    def score_completion_only(self, prompt_ids, prompt_mask, completion_ids, completion_mask):  
     
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)  
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  
        
        token_scores = self.forward_token_level(input_ids, attention_mask)  
        
        return token_scores[:, prompt_ids.shape[1]:, :]