import torch
import random
import re
import pandas as pd
import numpy as np
import os
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import evaluate
from tqdm import tqdm
import nltk
from rich.pretty import pprint

# Stop Warning
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--test-model-path', type=str, default="t5-small")
# parser.add_argument('--auto', type=bool, default=False, help="Auto regressive")
parser.add_argument('--cuda', type=str, default=0)
pargs = parser.parse_args()

# import wandb
os.environ["CUDA_VISIBLE_DEVICES"] = pargs.cuda
model_checkpoint = pargs.test_model_path

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = model.to("cuda")
model.eval()

data = [
    "In year 2011: Donald Trump is a member of the  ",
    "In year 2010: Donald Trump is a member of the  ",
]

for i in data:
    input_ids = tokenizer(i, return_tensors="pt").input_ids.to("cuda")
    prob = 1
    # outputs = model.generate(input_ids, max_new_tokens=10)
    print(input_ids)
    outputs = input_ids
    for _ in range(2):
        outputs = model(input_ids)

        # outputs = model(input_ids)
        outputs = torch.softmax(outputs["logits"][:, -1], dim=1)
        p, outputs = torch.max(outputs, dim=1)
        prob *= p

        # Append ouput to input
        input_ids = torch.cat([input_ids, outputs.unsqueeze(1)], dim=1)

    print("***************************************")
    # print(outputs)
    txt = tokenizer.decode(input_ids[0], skip_special_tokens=True) # Concatenated output
    print(f"\"{txt}\"", prob.item())
    print("***************************************")
    print("\n")
