from dataclasses import dataclass, field
from pathlib import Path
from trl import GRPOConfig

@dataclass  
class MaxEntIRLConfig(GRPOConfig):  
    """Configuration class for MaxEnt IRL Trainer."""  
    # Override GRPOConfig defaults  
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW"})  
    warmup_ratio: float = field(default=0.1, metadata={"help": "Linear warmup ratio"})  
    num_generations: int = field(default=8, metadata={"help": "Number of generations per prompt"})  
    max_completion_length: int = field(default=128, metadata={"help": "Maximum length of generated completion"})  
    per_device_train_batch_size: int = field(default=16, metadata={"help": "Batch size per device"})  
    shuffle_dataset: bool = field(default=False, metadata={"help": "Whether to shuffle training dataset"})  
    num_train_epochs: int = field(default=4, metadata={"help": "Total number of training epochs"})  
    num_iterations: int = field(default=1, metadata={"help": "Number of iterations per generation"})  
    beta: float = field(default=0.05, metadata={"help": "KL penalty coefficient"})  
    log_completions: bool = field(default=True, metadata={"help": "Whether to log completions"})  
    num_completions_to_print: int = field(default=2, metadata={"help": "Number of completions to print"})  
    save_strategy: str = field(default="steps", metadata={"help": "Save checkpoint strategy"})  
    save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X steps"})  
    report_to: str = field(default="wandb", metadata={"help": "Reporting tool"})  
    output_dir: str = field(default="./output", metadata={"help": "Output directory"})  
      
    # MaxEnt IRL specific parameters  
    reward_learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate for reward model optimizer"})  
    reward_max_length: int = field(default=256, metadata={"help": "Maximum sequence length for reward model tokenization"})  
    expert_batch_size: int = field(default=8, metadata={"help": "Batch size for computing expert features"})  
    reward_update_frequency: int = field(default=1, metadata={"help": "Frequency of reward model updates"})  
    reward_output_dir: str | None = field(default=None, metadata={"help": "Output directory for reward model checkpoints"})  
      
    def __post_init__(self):  
        super().__post_init__()  
        if self.reward_output_dir is None:  
            self.reward_output_dir = str(Path(self.output_dir) / "reward_model")

@dataclass  
class RSGenerationConfig:    
    num_candidate_tokens: int = field(default=3, metadata={"help": "Number of candidate tokens to generate at each step"})  
    max_new_tokens: int = field(default=128, metadata={"help": "Maximum number of tokens to generate"})  
    reward_weight: float = field(default=1.0, metadata={"help": "Weight for reward scores in token selection"})  
    temperature: float = field(default=0.7, metadata={"help": "Sampling temperature"})  
    do_sample: bool = field(default=True, metadata={"help": "Whether to use sampling or greedy decoding"})  
    device: str = field(default="cuda", metadata={"help": "Device for inference"})

@dataclass
class ModelArguments:
    # Target model (dual role) 
    target_model_id: str = field(default="/home/qjm/my-model/Llama-2-7b-chat-hf", metadata={"help": "Target aligned model to attack."})
    policy_model_id: str = field(default="/home/qjm/my-model/Llama-2-7b-hf", metadata={"help": "Policy model id or path"})  
    reward_model_id: str = field(default="Skywork/Skywork-Reward-V2-Llama-3.2-1B", metadata={"help": "Reward model id or path"}) 
    # Reward head path (for loading trained reward model)  
    reward_head_path: str | None = field(default=None, metadata={"help": "Path to trained reward head checkpoint"})  
    # Peft config
    use_peft: bool = field(default=True, metadata={"help": "Whether to use PEFT"})  
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})  
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})  
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})  

@dataclass
class DataArguments:
    prompt_dataset_name: str = field(default="Anthropic/hh-rlhf", metadata={"help": "Prompt dataset name"})  
    prompt_dataset_config: str = field(default="standard_prompt_only", metadata={"help": "Prompt dataset config"})  
    expert_dataset_name: str = field(default="Anthropic/hh-rlhf", metadata={"help": "Expert dataset name"})  
    expert_dataset_config: str = field(default="standard_prompt_completion", metadata={"help": "Expert dataset config"})  



















# @dataclass  
# class MaxEntIRLConfig(GRPOConfig):  
#     """Configuration class for MaxEnt IRL Trainer."""   
#     reward_learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate for reward model optimizer"})  
#     reward_max_length: int = field(default=256, metadata={"help": "Maximum sequence length for reward model tokenization"})   
#     expert_batch_size: int = field(default=8, metadata={"help": "Batch size for computing expert features during initialization"})  
#     reward_update_frequency: int = field(default=1, metadata={"help": "Frequency of reward model updates"})  
#     reward_output_dir: str | None = field(default=None, metadata={"help": "Output directory for reward model checkpoints"})  

#     def __post_init__(self):  
#         super().__post_init__()  
#         if self.reward_output_dir is None:  
#             self.output_dir = "./output"
#             self.reward_output_dir = str(Path(self.output_dir) / "reward_model")
#             self.learning_rate = 5e-5
#             self.warmup_ratio = 0.1
#             self.num_generations = 8
#             self.max_completion_length = 128  
#             self.per_device_train_batch_size = 16
#             self.shuffle_dataset = False
#             self.num_train_epochs = 4
#             self.num_iterations = 1
#             self.beta = 0.05
#             self.log_completions = True
#             self.num_completions_to_print = 2
#             self.save_strategy = "steps"
#             self.save_steps = 100  
#             self.report_to = "wandb"

# from transformers import TrainingArguments
# @dataclass
# class IRLArguments(TrainingArguments):
#     # num_iterations: int = field(default=10, metadata={"help": "Number of training iterations"})
#     output_dir: str = field(default="./output", metadata={"help": "The output directory where the model checkpoints will be stored."})
#     # Policy
#     policy_lr: float = field(default=5e-5, metadata={"help": "The initial learning rate for policy model training."})
#     num_generations: int | None = field(default=8, metadata={"help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size * gradient_accumulation_steps) must be evenly divisible by this value."})
#     max_completion_length: int | None = field(default=128, metadata={"help": "Maximum length of the generated completion."})
#     # Reward
#     reward_lr: float = field(default=1e-5, metadata={"help": "The initial learning rate for reward model training."})
#     max_length: int | None = field(default=256, metadata={"help": "Maximum length of the full text."})

#     def __post_init__(self):  
#         super().__post_init__()   
#         self.policy_output_dir = Path(self.output_dir) / "policy_output"  
#         self.reward_output_dir = Path(self.output_dir) / "reward_output"  
#         self.per_device_train_batch_size = 16
#         self.num_train_epochs = 4