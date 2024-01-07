import random
import re
import pandas as pd
import numpy as np
import os

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Inference Mistral-7B')
parser.add_argument('--dataset-path', type=str, required=True, help='train dataset: Folder path containing csvs')
parser.add_argument('--model-name', type=str, default='mistralai/Mistral-7B-v0.1', help='model name')
parser.add_argument('--cuda', type=str, default='3', help='cuda')

args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def seed_everything(seed):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(1)
model_name = args.model_name # 'mistralai/Mistral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

tokenizer.pad_token = "[PAD]"
tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda",
    pad_token_id=tokenizer.eos_token_id
)

data = pd.DataFrame()
csvs = os.listdir(args.dataset_path)
csvs = [x for x in csvs if x.endswith(".csv")]
csvs = sorted(csvs)
for csv in csvs:
    name = csv.split(".")[0]
    print("Loading", csv)
    temp = pd.read_csv(os.path.join(args.dataset_path, csv))
    data = pd.concat([data, temp], ignore_index=True)

predictions = []
for i in tqdm(range(data.shape[0])):
    prompt = data.iloc[i]['query']
    # print(prompt)
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=100, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.95,
        num_return_sequences=1,
    )
    predictions.append(sequences[0]['generated_text'])

data['prediction'] = predictions
data.to_csv("./results/mistralai/out.csv", index=False)

