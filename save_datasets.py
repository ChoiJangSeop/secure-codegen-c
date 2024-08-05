import json
from datasets import Dataset, load_dataset

c_fixes_json = []
n_fixed_file = 17326

for i in range(1,n_fixed_file+1):
    with open(f"c_fixes_raw/c{i}.c", 'r') as file:
        content = file.read()
        c_fixes_json.append({
            "label": content,
            "input_ids": content
        })

with open("./c_fixes.jsonl", 'w') as new_file:
    json.dump(c_fixes_json, new_file, indent=4)
    
    
dataset = load_dataset("json", data_files="./c_fixes.jsonl", split="train")
print(dataset)

# codegen-2B trainable params: 2,621,440 || all params: 2,781,977,600 || trainable%: 0.09422937122139301
