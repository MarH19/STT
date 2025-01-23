import os
import torch
import evaluate
from datasets import load_dataset, load_from_disk
from transformers import ( 
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    )
from transformers import EarlyStoppingCallback
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from config import Config
# py7zr for samsum dataset
# bitsandbytes also needed
# pip install rouge_score also needed
# need raise access for meta llama 3.2 model

class Finetuner():
    def __init__(self, model_name, dataset_name,finetuned_model):
        self.adapted_model = "temp"
        self.config = Config()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.final_model = finetuned_model
        self.compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = self.config.use_4bit,
            bnb_4bit_quant_type = self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype = self.compute_dtype,
            bnb_4bit_use_double_quant = self.config.use_nested_quant
            )
        self.rouge_metric = self.rouge_metric = evaluate.load("rouge")
    
    def bf16_check(self):
        '''
        Check if bf16 training is possible with the current GPU
        '''
        if self.compute_dtype == torch.float16 and self.config.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return True
            else:
                return False
        else: 
            return False
    
    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config = self.bnb_config,
            device_map = self.config.device_map,
        )

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1 # disable tensor parallelism

        #Load LLama tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code = True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    
    def configure(self):
        
        #Load QLoRA config
        self.peft_config = LoraConfig(
            lora_alpha = self.config.lora_alpha,
            lora_dropout = self.config.lora_dropout,
            r  = self.config.lora_r,
            bias = "none",
            task_type = "CAUSAL_LM",
        )

        #Set Training parameters
        self.training_arguments = TrainingArguments(
            output_dir = self.config.output_dir,
            num_train_epochs = self.config.num_train_epochs,
            per_device_train_batch_size = self.config.per_device_train_batch_size,
            per_gpu_eval_batch_size= self.config.per_device_eval_batch_size,
            gradient_accumulation_steps = self.config.gradient_accumulation_steps,
            optim = self.config.optimizer,
            #save_steps = self.config.save_steps,  
            save_strategy="steps", # save_strategy and eval_strategy must be identic
            logging_steps = self.config.logging_steps,
            learning_rate = self.config.learning_rate,
            fp16 = self.config.fp16,
            bf16 = self.bf16_check(),
            max_grad_norm = self.config.max_grad_norm,
            weight_decay = self.config.weight_decay,
            lr_scheduler_type = self.config.lr_scheduler_type,
            warmup_ratio = self.config.warmup_ratio,
            group_by_length = self.config.group_by_length,
            max_steps = self.config.max_steps,
            report_to = "tensorboard",
            eval_strategy ="steps",
            eval_steps=self.config.eval_steps, # in config anpassen
            load_best_model_at_end=False,
           
        )

    def load_dataset(self, name="prepared_samsum_dataset", num_samples:int = -1):
        #dataset = load_dataset(self.dataset_name, split = "train", trust_remote_code=True)
        #eval_dataset = load_dataset(self.dataset_name, split="validation", trust_remote_code=True)
        
        dataset = load_from_disk(r"C:\Users\marco\master_thesis\samsum\train")
        eval_dataset = load_from_disk(r"C:\Users\marco\master_thesis\samsum\validation")
        # to select a subset
        if num_samples != -1:
            dataset = dataset.select(range(num_samples)) 
            eval_dataset = eval_dataset.select(range(num_samples // 10))

        def format_example(example):
            messages = [
                {"role": "system", "content": f"you are helpful and creative assistant which can create dialogues based on summaries of dialogues"},
                {"role": "user", "content": f"Generate a dialogue based on the following summary:\n\n{example['summary']}"},
                {"role": "assistant", "content": example['dialogue']}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return {"text": prompt}
            

        self.formatted_dataset = dataset.map(format_example)
        self.validation_dataset = eval_dataset.map(format_example)
        #self.formatted_dataset = dataset.map(format_example, remove_columns=["id", "summary", "dialogue"])
        #self.validation_dataset = eval_dataset.map(format_example, remove_columns=["id", "summary", "dialogue"])
        self.formatted_dataset.save_to_disk(name + "_train")
        self.validation_dataset.save_to_disk(name + "_eval")
    
    def compute_metrics(self, eval_preds):
        """
        Computes ROUGE-1 and ROUGE-L as metrics.
        """
        preds, labels = eval_preds

        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a list of strings
        result = self.rouge_metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract ROUGE-1 and ROUGE-L scores
        rouge_1 = result["rouge1"].mid.fmeasure
        rouge_l = result["rougeL"].mid.fmeasure

        return {
            "rouge-1": rouge_1,
            "rouge-L": rouge_l
        }


    def train(self):
        # SFT Trainer
        self.trainer = SFTTrainer(
            model = self.model,
            train_dataset = self.formatted_dataset,
            eval_dataset= self.validation_dataset,
            peft_config = self.peft_config,
            args = self.training_arguments,
            tokenizer = self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        

        # Create early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,  # Number of evaluations to wait before stopping
            early_stopping_threshold=0.01  # Minimum improvement to qualify as better
        )

        # Add the callback to the trainer
        self.trainer.add_callback(early_stopping)
        # Start training
        self.trainer.train()

        # save trained model
        self.trainer.model.save_pretrained(self.adapted_model)
    
    def save_model(self):
        
        # Ignore warnings
        logging.set_verbosity(logging.CRITICAL)

        # Reload model in FP16 and merge it with LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=self.config.device_map,
        )
        model = PeftModel.from_pretrained(base_model, self.adapted_model)
        model = model.merge_and_unload()

        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # check if the final model path exists or not
        os.makedirs(self.final_model, exist_ok=True)

        model.save_pretrained(self.final_model)
        tokenizer.save_pretrained(self.final_model)

        print(f"Final merged model and tokenizer saved to {self.final_model}")
    def run_pipeline(self, num_samples=-1):
        # configure pipeline arguments
        self.configure()
        # load the model and tokenizer
        self.load()

        # extract the last name of the datasetpath if existing
        if '/' in self.dataset_name:
            name = self.dataset_name.split('/')[-1]
        else:
            name = self.dataset_name
        
        # load and modify the dataset
        self.load_dataset(name,num_samples)

        # start the training
        self.train()

        # save the newly trained model 
        self.save_model()


if __name__ == "__main__":
   
    #model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model_name = r"C:\Users\marco\master_thesis\llama_3_2_1B_instruct_local"
    dataset_name = "Samsung/samsum"
    new_model = "fine_tuned_model"
    finetuner = Finetuner(model_name,dataset_name,new_model)
    finetuner.run_pipeline()



