# set-up for Lora fine-tuning
import json
import torch
import argparse
import deepspeed
import pandas as pd
from deepspeed import get_accelerator

from torch.utils.data import DataLoader, TensorDataset 
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorWithPadding, 
    TrainingArguments, 
    Trainer
)

from peft import get_peft_model, LoraConfig, TaskType

class CustomDataset(Dataset):
    def __init__(self, data):
        self._data = data
        
    def __len__(self):
        return len(self._data)
    def __getitem__(self, idx):
        print("[custom datset] idx: ", idx)
        if isinstance(idx, list): idx = idx[0]
        print("[custom datset - parsing] idx: ", idx)
        return (
            torch.LongTensor(self._data[idx]['input_ids']), 
            torch.LongTensor(self._data[idx]['input_ids'])
        )

# model_path = "DDIDU/ETRI_CodeLLaMA_7B_CPP"
model_path = "Salesforce/codegen-350M-multi"
dataset_path = "./c_fixes.json"

# with open(dataset_path, 'r') as file:
#     c_fixes_json = json.load(file)

with open('./c_fixes.jsonl', 'r') as file:
    data = json.load(file)

# dataset = load_dataset("json", data_files="./c_fixes.json", split="train")
dataset = Dataset.from_pandas(pd.DataFrame(data))
# dataset = load_dataset('imdb')

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_funtion(examples):
    return tokenizer(examples['input_ids'], max_length=512, padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_funtion, batched=True)
tokenized_datasets.set_format("torch")
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

# train configuration (with DeepSpeed)


# ds_dataset = CustomDataset(tokenized)
data_loader = DataLoader(tokenized_datasets, batch_size=16, shuffle=True)

ds_config = {
    "train_batch_size": 4,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {"lr": 0.001}
    },
    "zero": {
        "stage": 3,
        "offload_optimizer": {
            "device": "[cpu|nvme]"
        },
        "offload_param": {
            "device": "[cpu|nvme]"
        }
    },
    "fp16": {
        "enabled": True
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    training_data=tokenized_datasets,
    config=ds_config
)

cnt = 1
for batch in data_loader:
    #forward() method
    outputs = model_engine(**batch)

    #runs backpropagation
    model_engine.backward(outputs.loss)

    #weight update
    model_engine.step()
    cnt += 1
    print("step # ", cnt)
    
