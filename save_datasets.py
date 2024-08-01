import json
from datasets import Dataset, load_dataset

c_fixes_json = []
n_fixed_file = 17326

for i in range(1,n_fixed_file+1):
    with open(f"c_fixes_raw/c{i}.c", 'r') as file:
        content = file.read()
        c_fixes_json.append({
            "idx": str(i),
            "content": content
        })

with open("./c_fixes.json", 'w') as new_file:
    json.dump(c_fixes_json, new_file, indent=4)
    
    
dataset = load_dataset("json", data_files="./c_fixes.json", split="train")
print(dataset)
