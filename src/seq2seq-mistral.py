import random
import re
import pandas as pd
import numpy as np
import os

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse

parser = argparse.ArgumentParser(description='Finetune Mistral-7B')
parser.add_argument('--model-name', type=str, default='mistralai/Mistral-7B-v0.1', help='model name')
parser.add_argument('--base-model-id', type=str, default='mistralai/Mistral-7B-v0.1', help='base model id')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--batch-size', type=int, default=2, help='train batch size')
parser.add_argument('--dataset-path', type=str, required=True, help='train dataset: Folder path containing csvs')
parser.add_argument('--num', type=bool, default=False, help='numerical or non-numerical')
parser.add_argument('--cuda', type=str, default='3', help='cuda')

args = parser.parse_args()

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
model_name = args.model_name # 'mistralai/Mistral-7B-v0.1'
base_model_id = args.base_model_id # "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

from datasets import load_dataset, Dataset, DatasetDict

csvs = os.listdir(args.dataset_path)
csvs = [x for x in csvs if x.endswith(".csv")]
csvs = sorted(csvs)
if "yearly_freq.csv" in csvs:
    csvs.remove("yearly_freq.csv")

data = pd.DataFrame()
for csv in csvs:
    name = csv.split(".")[0]
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

def tokenize(prompts):
    # TODO: Add Preprocessing of prompts here
    # print(prompts)
    # exit()
    prompts = [doc.strip() + " " + str(ans).strip() for doc, ans in zip(prompts["query"], prompts["answer"])]
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

# def generate_and_tokenize_prompt(data_point):
#     full_prompt =f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
#     This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
#     The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']


#     ### Target sentence:
#     {data_point["target"]}



#     ### Meaning representation:
#     {data_point["meaning_representation"]}
#     """
#     return tokenize(full_prompt)
# tokenizer.pad_token = "[PAD]"
# tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
# tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

# print(tokenized_train_dataset[4]['input_ids'])
# print(len(tokenized_train_dataset[4]['input_ids']))

# print("Target Sentence: " + test_dataset[1]['target'])
# print("Meaning Representation: " + test_dataset[1]['meaning_representation'] + "\n")

# eval_prompt = """Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
# This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
# The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']


# ### Target sentence:
# Earlier, you stated that you didn't have strong feelings about PlayStation's Little Big Adventure. Is your opinion true for all games which don't have multiplayer?


# ### Meaning representation:
# """

# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
# model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True))

# from peft import prepare_model_for_kbit_training
# model.gradient_checkpointing_enable()
# model = prepare_model_for_kbit_training(model)

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
# Apply the accelerator. You can comment this out to remove the accelerator.
# model = accelerator.prepare_model(model)

print(model)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True


import transformers
from datetime import datetime


project = "tme-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = f"./results/{model_name}-{ft_type}-finetuned-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
os.makedirs(output_dir, exist_ok=True)
print("Saving to", output_dir)



print("***"*50)

training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=3e-4, # Want about 10x smaller than the Mistral learning rate
        logging_steps=50,
        optim="paged_adamw_8bit",
        logging_dir=output_dir,        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,
        save_total_limit=2,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="tensorboard",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    )

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_temporal_dataset["train"],
    eval_dataset=tokenized_temporal_dataset["validation"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_id,  # Mistral, same as before
    
#     device_map="cuda",
#     trust_remote_code=True,
#     use_auth_token=True
# )
# tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token


# from peft import PeftModel
# ft_model = PeftModel.from_pretrained(base_model, "mistral-viggo-finetune/checkpoint-1000")

# ft_model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100, pad_token_id=2)[0], skip_special_tokens=True))


############### 
# Reference
############### https://www.datacamp.com/tutorial/mistral-7b-tutorial