# set-up for Lora fine-tuning
import json
import torch
import argparse
import deepspeed
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
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

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

# train configuration (with DeepSpeed)

ds_dataset = CustomDataset(tokenized['input_ids'], tokenized['input_ids'])
data_loader = DataLoader(ds_dataset)

ds_config = {
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
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    training_data=ds_dataset,
    config=ds_config
)


for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()
    
