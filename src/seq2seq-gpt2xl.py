import random
import re
import pandas as pd
import numpy as np
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_dataset, Dataset, DatasetDict
import torch
import argparse
from rich.pretty import pprint

parser = argparse.ArgumentParser(description='Finetune GPT2XL')
parser.add_argument('--dataset-path', type=str, required=True, help='train dataset: Folder path containing csvs')
parser.add_argument('--start-year', type=int, default=1947, help='start year')
parser.add_argument('--end-year', type=int, default=2020, help='end year')
parser.add_argument('--model-name', type=str, default='gpt2-xl', help='model name')
parser.add_argument('--base-model-id', type=str, default='gpt2-xl', help='base model id')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--batch-size', type=int, default=2, help='train batch size')
parser.add_argument('--num', type=bool, default=False, help='numerical or non-numerical')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--save-limit', type=int, default=1, help='save limit for checkpoints')
parser.add_argument('--patience', type=int, default=3, help='Early Stopping patience')
parser.add_argument('--prefix', type=str, default='Genereate only the number for the following query ', help='prefix')
parser.add_argument('--cuda', type=str, default='3', help='cuda')

args = parser.parse_args()
pprint(args)

################### Setting up ###################
ft_type = "numerical" if args.num else "non-numerical"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
os.environ["TOKENIZERS_PARALLELISM"]="false"

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
seed_everything(args.seed)

################### Setting up models ###################
model_name = args.model_name # 'mistralai/Mistral-7B-v0.1'
base_model_id = args.base_model_id # "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        trust_remote_code=True,
    )

################### Loading Dataset ###################
csvs = os.listdir(args.dataset_path)
csvs = [x for x in csvs if x.endswith(".csv")]
csvs = sorted(csvs)
if "yearly_freq.csv" in csvs:
    csvs.remove("yearly_freq.csv")

data = pd.DataFrame()
for csv in csvs:
    name = csv.split(".")[0]
    if args.start_year <= int(name) <= args.end_year:
        print("Loading", csv)
        temp = pd.read_csv(os.path.join(args.dataset_path, csv))
        data = pd.concat([data, temp], ignore_index=True)

data = data.astype(str)

if "year" in data.columns:
    data.drop(columns=["year"], inplace=True)
if "frequency" in data.columns:
    data.drop(columns=["frequency"], inplace=True)

data = data.sample(frac=1, random_state=args.seed).reset_index(drop=True)
eval_data = data.sample(frac=0.2, random_state=args.seed).reset_index(drop=True)

train_dataset = Dataset.from_pandas(data)
eval_dataset = Dataset.from_pandas(eval_data)

temporal_dataset = DatasetDict({"train": train_dataset, "validation": eval_dataset})


# train_dataset = load_dataset('gem/viggo', split='train')
# eval_dataset = load_dataset('gem/viggo', split='validation')
# test_dataset = load_dataset('gem/viggo', split='test')

print(train_dataset)
print(eval_dataset)
# print(test_dataset)

################### Tokenization and pre-processing prompts ###################
def tokenize(prompts):
    # TODO: Add Preprocessing of prompts here
    # print(prompts)
    # exit()
    if args.prefix != "":
        prefix = args.prefix.strip() + " "
    prompts = [prefix + doc.strip() + " " + str(ans).strip() for doc, ans in zip(prompts["query"], prompts["answer"])]
    result = tokenizer(
        prompts,
        truncation=True,
        max_length=56,
        padding="max_length",
    )
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    result["labels"] = result["input_ids"].copy()
    return result

tokenizer.pad_token = "[PAD]"
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizing...")
tokenized_temporal_dataset = temporal_dataset.map(tokenize, batched=True, batch_size=1000)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



print_trainable_parameters(model)

print(model)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True


project = "tme-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = f"./results/{base_model_id}-{ft_type}-finetuned-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
os.makedirs(output_dir, exist_ok=True)
print("Saving to", output_dir)



print("***"*50)

training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size= args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr, # Want about 10x smaller than the Mistral learning rate
        logging_steps=50,
        optim="paged_adamw_8bit",
        logging_dir=output_dir,        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,
        save_total_limit=args.save_limit,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="tensorboard",           # Comment this out if you don't want to use weights & baises
        load_best_model_at_end=True,
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_temporal_dataset["train"],
    eval_dataset=tokenized_temporal_dataset["validation"],
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


############### 
# Reference
############### https://www.datacamp.com/tutorial/mistral-7b-tutorial