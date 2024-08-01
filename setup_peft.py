# set-up for Lora fine-tuning
import json
import torch
import argparse
import deepspeed

from torch.utils.data import DataLoader, TensorDataset 
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

# train configuration (with DeepSpeed)
tokenized.set_format('torch')
input_ids = tokenized['input_ids']
labels = tokenized['input_ids']

dataset = TensorDataset(input_ids, labels)
data_loader = DataLoader(dataset, batch_size=8)

parser = argparse.ArgumentParser(description="traing scripts")
parser.add_argument('--local-rank', type=int, default=-1, help='local rank passed from distributed launcher')

parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args()

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=cmd_args,
    model=model,
    model_parameters=model.parameters(),
    training_data=tokenized
)

for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()
    
