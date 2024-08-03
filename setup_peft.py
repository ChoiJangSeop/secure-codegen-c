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

dataset = load_dataset("json", data_files="./c_fixes.json", split="train")

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_funtion(examples):
    return tokenizer(examples['content'], max_length=512, padding="max_length", truncation=True)

tokenized = dataset.map(tokenize_funtion, batched=True)
print(tokenized.column_names)

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
    gradient_accumulation_steps=1,
    logging_dir='./logs',  # Optional: logs directory
    logging_steps=10,      # Optional: log every N steps
    evaluation_strategy="epoch",  # Optional: evaluation strategy
    save_strategy="epoch",  # Optional: save model every epoch
    deepspeed="ds_falcon_180b_z3.json"  # DeepSpeed 설정 파일
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

trainer.train()



