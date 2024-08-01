# set-up for Lora fine-tuning
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorWithPadding, 
    TrainingArguments, 
    Trainer
)

from peft import get_peft_model, LoraConfig, TaskType

model_path = "DDIDU/ETRI_CodeLLaMA_7B_CPP"
dataset_path = "./c_fixes.json"

# with open(dataset_path, 'r') as file:
#     c_fixes_json = json.load(file)

dataset = load_dataset("json", data_files="./c_fixes.json", split="train")

device_map = {"": torch.cuda.current_device()}
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_funtion(examples):
    return tokenizer(examples['content'], max_length=512, padding="max_length", truncation=True)

tokenized = dataset.map(tokenize_funtion, batched=True)
# tokenized['label'] = tokenized['input_ids']

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# train configuration

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./result",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1
)



trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

with torch.autocast("cuda"):
    trainer.train()



