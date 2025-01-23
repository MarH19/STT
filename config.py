from dataclasses import dataclass

@dataclass
class Config:
    # LoraConfig parameters
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout:int = 0.1

    # BitsAndBytesConfig parameters
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16" # data type during training 
    bnb_4bit_quant_type: str = "nf4"  # quantization type
    use_nested_quant: bool = True # double quantization

    # Trainingarguments (transformers)
    output_dir: str = "./results" # output directory where the model predictions and checkpoints will be stored
    num_train_epochs: int = 2
    fp16: bool = False # enable fp16/bf16 training 
    bf16: bool = True # only possible with specific GPU
    per_device_train_batch_size: int = 4 # batch size per GPU for training
    per_device_eval_batch_size: int = 4 # batch size per GPU for evaluation
    gradient_accumulation_steps: int = 4 # gradient accumulation steps - No of update steps
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    max_grad_norm = 0.3 # gradient clipping (max gradient Normal)
    optimizer: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    seed: int = 1
    max_steps: int = -1
    warmup_ratio: float = 0.03 # ratio of steps for linear warmup
    group_by_length: bool = True # group sequnces into batches with same length
    save_steps: int = 0 # save checkpoint every X updates steps
    logging_steps: int = 50 # log at every X updates steps
    eval_steps: int = 50

    device_map = {"":0}
    
    
    


