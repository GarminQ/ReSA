from typing import Optional  
import os 
import torch 

from torch import nn
from transformers import PreTrainedModel
from peft import PeftConfig
from trl import GRPOTrainer  

from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.data_utils import (
    apply_chat_template,
    is_conversational
)

from pandora.models import CustomRewardModel
from pandora.arguments import MaxEntIRLConfig

logger = logging.get_logger(__name__)

class MaxEntIRLTrainer(GRPOTrainer):  
    def __init__(  
        self,  
        args: MaxEntIRLConfig,
        policy_model: PreTrainedModel, 
        reward_model: CustomRewardModel,
        peft_config: PeftConfig, 
        train_dataset=None,  
        eval_dataset=None,  
        expert_dataset=None,  
        processing_class=None, 
        reward_processing_classes=None, 
        callbacks=None,  
        optimizers=(None, None),  
    ):  
        """  
        Initialize MaxEnt IRL Trainer combining GRPO policy optimization with reward learning.  
        
        Args:  
            args: MaxEntIRLConfig containing all training parameters
            policy_model: Policy model to train (causal LM)  
            reward_model: Custom reward model for computing rewards   
            peft_config: PEFT config (e.g., LoRA) for parameter-efficient fine-tuning  
            train_dataset: Training dataset with prompt column  
            eval_dataset: Evaluation dataset  
            expert_dataset: Expert demonstrations with prompt and response columns  
            processing_class: Tokenizer/processor for policy model  
            reward_processing_classes: List of tokenizers for reward model  
            callbacks: Training callbacks  
            optimizers: (optimizer, scheduler) tuple  
        """
        super().__init__(  
            args=args, 
            model=policy_model,  
            reward_funcs=reward_model, 
            train_dataset=train_dataset,  
            eval_dataset=eval_dataset,  
            processing_class=processing_class,  
            reward_processing_classes=reward_processing_classes,   
            callbacks=callbacks,  
            optimizers=optimizers,  
            peft_config=peft_config,  
        )  
          
        self.reward_model = reward_model 
        self.reward_tokenizer = self.reward_processing_classes[0] 
        
        self.reward_optimizer = torch.optim.Adam(  
            self.reward_model.score_head.parameters(),   
            lr=args.reward_learning_rate  
        )  
         
        self.expert_features = self._compute_expert_features(  
            expert_dataset,  
            batch_size=args.expert_batch_size  
        )  
      
    @torch.no_grad()  
    def _compute_expert_features(self, expert_dataset, batch_size):  
        """  
        Precompute expert trajectory features 𝔼[f(ζ_expert)].  
        
        Extracts feature representations from expert demonstrations once at initialization  
        to avoid redundant computation during training.  
        
        Args:  
            expert_dataset: Dataset with 'prompt' and 'response' columns  
            
        Returns:  
            list: Feature vectors for each expert demonstration  
        """
        self.reward_model.eval()  

        expert_dataset = expert_dataset.map(  
            lambda example: {  
                "messages": [  
                    {"role": "user", "content": example['prompt']},  
                    {"role": "assistant", "content": example['response']}  
                ]  
            }  
        )  
          
        all_features = []  

        for i in range(0, len(expert_dataset), batch_size):  
            expert_messages = expert_dataset[i:i + batch_size]["messages"]  
            formatted_batch = self.reward_tokenizer.apply_chat_template(expert_messages, tokenize=False)
            inputs = self.reward_tokenizer(  
                formatted_batch,  
                padding=True,  
                add_special_tokens=False,  
                return_tensors="pt"  
            ).to(self.reward_model.device)  
              
            features = self.reward_model.get_last_hidden_state(**inputs)  
            all_features.extend(features)  
          
        return all_features  

    # def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):  
    #     """  
    #     Compute rewards and update reward model at appropriate intervals.  
        
    #     Overrides parent GRPOTrainer to update reward model before computing rewards,  
    #     implementing MaxEnt IRL's alternating optimization of policy and reward model.  
        
    #     Args:  
    #         inputs: Input batch with prompt and other columns  
    #         prompts: List of prompt texts  
    #         completions: List of generated completion texts  
    #         completion_ids_list: List of completion token IDs  
            
    #     Returns:  
    #         torch.Tensor: Rewards per function per completion, shape (batch_size * num_generations, num_reward_funcs)  
    #     """ 
        
    #     if self.state.global_step % self.args.num_iterations == 0:
    #         self._update_reward_model(inputs=inputs, completions=completions)  
        
    #     return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
      
    def _update_reward_model(self, inputs, completions):  
        """  
        Update reward model using MaxEnt IRL algorithm.  
        
        Matches feature expectations between expert and policy trajectories:  
            ∇L = 𝔼[f(ζ_expert)] - 𝔼[f(ζ_policy)]  
        
        Args:  
            inputs: Current batch input data with query field  
            completions: Policy-generated completion texts  
            
        Note:  
            Uses gradient ascent by directly setting .grad attribute with negative gradient.  
        """  

        # Format policy trajectories as conversational messages for reward model 
        policy_messages = [[  
            {"role": "user", "content": input['query']},  
            {"role": "assistant", "content": completion}  
        ] for input, completion in zip(inputs, completions)]  
        formatted_batch = self.reward_tokenizer.apply_chat_template(policy_messages, tokenize=False)  
        batch_inputs = self.reward_tokenizer(  
            formatted_batch,  
            padding=True,  
            add_special_tokens=False,   
            return_tensors="pt"  
        ).to(self.reward_model.device)  
        
        # Extract policy trajectory features 𝔼[f(ζ_policy)]
        with torch.no_grad():  
            batch_policy_features = self.reward_model.get_last_hidden_state(**batch_inputs)  
        
        # Retrieve corresponding expert features for current batch
        batch_size = len(formatted_batch) // self.args.num_generations  
        step_in_epoch = int((self.state.global_step) % (self.state.max_steps / self.state.num_train_epochs)) 
        index = (step_in_epoch // self.args.num_iterations) * batch_size
        batch_expert_features = self.expert_features[index:index + batch_size]  
          
        # Compute MaxEnt IRL gradient: ∇L = 𝔼[f(ζ_expert)] - 𝔼[f(ζ_policy)]  
        policy_feature_mean = torch.mean(batch_policy_features, dim=0, keepdim=True)  
        expert_feature_mean = torch.mean(torch.stack(batch_expert_features), dim=0, keepdim=True)  
        gradient = expert_feature_mean - policy_feature_mean  
          
        # Update reward model using gradient ascent 
        self.reward_optimizer.zero_grad()  
        self.reward_model.score_head.weight.grad = -gradient  
        self.reward_optimizer.step()  

        # Log reward gradient norm for monitoring training progress
        if hasattr(self, '_metrics') and 'train' in self._metrics:  
            self._metrics["train"]["reward_grad_norm"] = [torch.norm(gradient).item()]  
      
    def _save_checkpoint(self, model, trial):  
        """  
        Save checkpoint including both policy and reward models.  
        
        Extends parent method to additionally save reward model state since MaxEnt IRL  
        trains two models simultaneously.  
        
        Args:  
            model: Policy model (saved by parent)  
            trial: Optuna trial object (optional, for hyperparameter search)  
            
        Saves:  
            - Policy model and optimizer (via super())  
            - Reward model score_head weights  
            - Reward optimizer state  
            - Reward model config  
        """   
        super()._save_checkpoint(model, trial)  
           
        if self.args.should_save and self.accelerator.is_main_process:
            checkpoint_dir = os.path.join(  
                self.args.output_dir,   
                f"checkpoint-{self.state.global_step}"  
            )  
            reward_model_dir = os.path.join(checkpoint_dir, "reward_model")  
            os.makedirs(reward_model_dir, exist_ok=True)  
                
            torch.save(  
                self.reward_model.score_head.state_dict(),  
                os.path.join(reward_model_dir, "score_head.pt")  
            )  
            torch.save(  
                self.reward_optimizer.state_dict(),  
                os.path.join(reward_model_dir, "optimizer.pt")  
            )  
              
            if hasattr(self.reward_model, 'config'):  
                self.reward_model.config.to_json_file(  
                    os.path.join(reward_model_dir, "config.json")  
                )

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        # TODO
        if self.state.global_step % self.args.num_iterations == 0:
            self._update_reward_model(inputs=inputs, completions=completions)  

        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    # TODO (check default padding_side="right")
                    reward_messages = [[  
                        {"role": "user", "content": input['query']},  
                        {"role": "assistant", "content": completion}  
                    ] for input, completion in zip(inputs, completions)]  
                    formatted_batch = self.reward_tokenizer.apply_chat_template(reward_messages, tokenize=False)  
                    reward_inputs = reward_processing_class(
                        text=formatted_batch, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func