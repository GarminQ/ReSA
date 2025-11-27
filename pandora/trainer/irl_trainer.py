from typing import Optional, Any 
import os 
import torch  

from transformers import PreTrainedModel
from peft import PeftConfig
from trl import GRPOTrainer  

from pandora.models import CustomRewardModel
from pandora.arguments import MaxEntIRLConfig

from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from torch import nn
from trl.extras.profiling import profiling_context, profiling_decorator

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
    prepare_multimodal_messages_vllm,
)

from trl.trainer.utils import (
    RepeatSampler,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    get_config_model_id,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)

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
        self.dense_rewards = True
        self.importance_sampling_level == "token"
      
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
            batch = expert_dataset[i:i + batch_size]["messages"]  
            formatted_batch = self.reward_tokenizer.apply_chat_template(batch, tokenize=False)  
            inputs = self.reward_tokenizer(  
                formatted_batch,  
                padding=True,  
                truncation=True,  
                max_length=self.args.reward_max_length,  
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
        batch_policy_message = [[  
            {"role": "user", "content": input['query']},  
            {"role": "assistant", "content": completion}  
        ] for input, completion in zip(inputs, completions)]  
        formatted_batch = self.reward_tokenizer.apply_chat_template(  
            batch_policy_message, tokenize=False  
        )  
        batch_inputs = self.reward_tokenizer(  
            formatted_batch,  
            padding=True,  
            truncation=True,  
            max_length=self.args.reward_max_length,  
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
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list, 
                           prompt_ids=None, prompt_mask=None, completion_ids=None, completion_mask=None):
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
                    # TODO
                    reward_processing_class = self.processing_class
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        if self.dense_rewards:
                            completions_socre = reward_func.score_completion_only(prompt_ids, prompt_mask, completion_ids, completion_mask)
                            completions_len = completions_socre.shape[1]
                            rewards_per_func = rewards_per_func.unsqueeze(-1).expand(-1, -1, completions_len).contiguous()
                            rewards_per_func[:, i, :] = completions_socre[:, :, 0]
                        else:
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

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
            self._generate(prompts)
        )

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template({"prompt": prompt}, self.processing_class, **self.chat_template_kwargs)["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                if isinstance(bootstrap, list):  # for VLM, the format might be [{"type": "text", "text": "..."}]
                    assert len(bootstrap) == 1 and bootstrap[0]["type"] == "text"
                    bootstrap = bootstrap[0]["text"]
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        # TODO
        # rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list, 
                                                    prompt_ids, prompt_mask, completion_ids, completion_mask)
        # TODO
        if not self.dense_rewards:
            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

            # Compute grouped-wise rewards
            num_generations = self.num_generations if mode == "train" else self.num_generations_eval
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards

            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll still log group level std
                std_rewards = rewards.view(-1, num_generations).std(dim=1)
                std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
            elif self.scale_rewards == "batch":
                # Compute global std
                std_rewards = rewards.std().expand_as(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)

        else:
            # rewards: [N, T] token-level rewards after weighting & summing over reward funcs
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0).unsqueeze(-1)
            ).nansum(dim=1)  # [N, T]
            num_generations = self.num_generations if mode == "train" else self.num_generations_eval

            # 1) take reward at the *last non-padding token* per completion (scalar per completion)
            last_tok_rewards = rewards[:, -1]  # [N]

            # 2) compute group-wise mean/std *across generations* using those last-token rewards only
            mean_last_rewards = last_tok_rewards.view(-1, num_generations).mean(dim=1) # [B]
            std_last_rewards = last_tok_rewards.view(-1, num_generations).std(dim=1) # [B] (population std)
           
            # 3) expand mean/std back to the flattened [N, 1] and broadcast over tokens
            mean_grouped_rewards = mean_last_rewards.repeat_interleave(num_generations, dim=0).unsqueeze(-1)  # [N, 1]
            std_rewards = std_last_rewards.repeat_interleave(num_generations, dim=0).unsqueeze(-1)    # [N, 1]
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

            # 4) normalise all token rewards by the last-token stats
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
        
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        # keep the aggregated advantages for logging
        all_process_advantages = advantages.clone()  
        all_process_advantages = all_process_advantages.mean(dim=-1) if self.dense_rewards else all_process_advantages
        
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            # self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            if not self.dense_rewards:
                self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            else:
                self._logs["rewards"][name].extend(rewards_per_func[:, i].mean(dim=-1).tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output
    
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
        # TODO
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss