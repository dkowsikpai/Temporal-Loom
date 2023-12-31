# -*- coding: utf-8 -*-
"""Seq2Seq_Himanshu.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ej9xfnsBIoMuLJgepZOcpXcJ5TVncmkl
"""

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

MASK_MAP = {
    "t5-base": "<extra_id_0>",
    "t5-large": "<extra_id_0>",
    "BART": "<mask>",
    "gpt2": "",
    "gpt2-large": "",
    "default": "<extra_id_0>"
}
    

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--val', type=str, required=True)
# parser.add_argument('--year', type=str, required=True, default="2010")
parser.add_argument('--model', type=str, default="t5-small")
parser.add_argument('--epoch', type=int, default=50)
# parser.add_argument('--sample', type=int, default=10)
parser.add_argument('--test-model', type=bool, default=False)
parser.add_argument('--test-model-path', type=str, default="t5-small")
parser.add_argument('--auto', type=bool, default=False, help="Auto regressive")
parser.add_argument('--postfix', type=str, default="2")
parser.add_argument('--cuda', type=str, default=0)
pargs = parser.parse_args()

# import wandb
os.environ["CUDA_VISIBLE_DEVICES"] = pargs.cuda
# os.environ["WANDB_PROJECT"]="generation_bot"

# save your trained model checkpoint to wandb
# os.environ["WANDB_LOG_MODEL"]="true"
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

# Auto Models
auto_model = ["gpt2", "gpt2-large", "gpt2-xl", "mistralai/Mistral-7B-v0.1"]

# Load the dataset:
model_checkpoint = pargs.model
mask = MASK_MAP.get(model_checkpoint, MASK_MAP["default"])

if model_checkpoint in [ "t5-base", "t5-large", "t5-3b", "t5-11b"]: # "gpt2", "gpt2-large", "gpt2-xl",
    prefix = "answer: "
else:
    prefix = ""

auto_reggressive = pargs.auto

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
if model_checkpoint in auto_model or auto_reggressive:
    print("Auto regressive")

    auto_reggressive = True

    if pargs.test_model:
        model_checkpoint = pargs.test_model_path

    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    def preprocess_function(examples):
        inputs = [doc.replace("_X_", "") + " " + ans for doc, ans in zip(examples["question"], examples["answer"])]
        model_inputs = tokenizer(inputs, max_length=max_target_length, truncation=True, padding=True)
        # pprint(inputs)
        # exit()

        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

else:
    if pargs.test_model:
        model_checkpoint = pargs.test_model_path
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)

    def preprocess_function(examples):
        inputs = [prefix + doc.replace("_X_", mask) for doc in examples["question"]]
        model_inputs = tokenizer(inputs, max_length=max_target_length, truncation=True)

        # # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["answer"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# tokenizer.save_pretrained(f"../temporal/results/pretrained-{model_checkpoint}")
# model.save_pretrained(f"../temporal/results/pretrained-{model_checkpoint}")
# print("here")
# exit()


data_path = pargs.train
reviews = pd.read_csv(data_path)
val_reviews = pd.read_csv(pargs.val)
print(reviews)
print(reviews.head())
reviews.info()

# Get Token length of each review
def get_token_length(review):
    # print(review)
    return tokenizer(review)['input_ids'].__len__()


# Take the average length of the reviews:
q_tokens_len = [get_token_length(review) for review in reviews['question']]
sum_all_tokens = sum(q_tokens_len)
avg_length_ques = sum_all_tokens / len(reviews['question'])
a_tokens_len = [get_token_length(review) for review in reviews['answer']]
sum_all_tokens = sum(a_tokens_len)
avg_length_ans = sum_all_tokens / len(reviews['answer'])
print("Avg length of question: ", avg_length_ques)
print("Avg length of answer: ", avg_length_ans)
print("Max length of question: ", max(q_tokens_len))
print("Max length of answer: ", max(a_tokens_len))

max_input_length = max(q_tokens_len) # max length of the input text
max_target_length = max(a_tokens_len) # max length of the target text

# raw_datasets = load_dataset("xsum")
print("Evaluation metric loaded")
metric = evaluate.load('rouge', "bleu")


ds_reviews = Dataset.from_pandas(reviews)
val_reviews = Dataset.from_pandas(val_reviews)

# train_testvalid = ds_reviews.train_test_split(shuffle = True, seed = 1, test_size=0.2)
# # Split the 10% test + valid in half test, half valid
# test_valid = train_testvalid['test'].train_test_split(shuffle = True, seed = 1, test_size=0.5)
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': ds_reviews,
    'test': val_reviews,
    'validation': val_reviews})



raw_datasets = train_test_valid_dataset


# block_size = 128
# def preprocess_function(examples):
#     return tokenizer([" ".join(x) + " " + " ".join(y) for x, y in zip(examples["question"], examples["answer"])], truncation=True)


# def group_texts(examples):
#     concatenated_examples = examples["input_ids"]
#     total_length = len(concatenated_examples)
#     result = {
#         "input_ids": [examples["input_ids"][i : i + block_size] for i in range(0, total_length, block_size)]
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result


print("Preprocessing function running")
preprocess_function(raw_datasets['train'][:2])

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
# lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# # # peint sample of tokenized dataset
# print("Sample of tokenized dataset")
# print(tokenized_datasets['train'][0])
# exit()

print ("Model Checkpoint: ", model_checkpoint)
# exit()


print("Setting up training arguments")
batch_size = 32
model_name = model_checkpoint.split("/")[-1]
os.makedirs(f"./logs/{model_name}-finetuned2-{pargs.postfix}", exist_ok=True)
args = Seq2SeqTrainingArguments(
    f"./results/{model_name}-finetuned2-{pargs.postfix}",
    evaluation_strategy = "epoch",
    logging_steps=100, 
    learning_rate=3e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=pargs.epoch,
    save_strategy="steps",
    eval_steps = 100,
    do_eval=True,
    do_train=True,
    predict_with_generate=True,
    seed=1,
    report_to = "tensorboard"
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # Extract a few results
    # result = {key: value for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

print("Setting up trainer")
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# trainer.push_to_hub(f"temporal/{model_name}-finetuned-tempwiki")
# exit()

if not pargs.test_model or pargs.epoch == 1:
    print("Training")
    trainer.train()
    trainer.save_model(f"./logs/{model_name}-finetuned2-{pargs.postfix}/checkpoint")
    # trainer.push_to_hub(f"temporal/{model_name}-finetuned-tempwiki")

# Generate some predictions
print("Generating predictions")

data_to_csv = []

our_metrics = {
    "accuracy": [],
    "exact_odering": [],
    "relaxed_ordering": [],
}

# from metrics import accuracy, exact_ordering, relaxed_ordering
if auto_reggressive:
    mask = ""

for i in tqdm(range(len(val_reviews))):
    # idx = random.randint(0, len(ds_reviews))
    idx = i
    question = prefix + val_reviews['question'][idx].replace("_X_", mask)
    # print(question)
    # exit()
    answer = val_reviews['answer'][idx]
    input_dict = tokenizer(question, return_tensors="pt")
    input_ids = input_dict["input_ids"].to("cuda")
    pred_ids = trainer.model.generate(input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id)
    # print(pred_ids)
    pred_answer = tokenizer.batch_decode(pred_ids.sequences, skip_special_tokens=True)[0]
    # print(pred_answer)
    # exit()

    transition_scores = model.compute_transition_scores(pred_ids.sequences, pred_ids.scores, normalize_logits=True)
    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = pred_ids.sequences[:, input_length:]

    # for tok, score in zip(generated_tokens[0], transition_scores[0]):
    #     # | token | token string | logits | probability
    #     score = score.to("cpu")
    #     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
    
    # Combined probability for the sequence
    # print(f"Probability: {np.exp(transition_scores[0].sum().cpu().numpy()):.2%}")

    # print(transition_scores)
    # exit()

    # acc = accuracy(pred_answer, answer)
    # eo = exact_ordering(pred_answer, answer)
    # ro = relaxed_ordering(pred_answer, answer)

    # our_metrics["accuracy"].append(acc)
    # our_metrics["exact_odering"].append(eo)
    # our_metrics["relaxed_ordering"].append(ro)

    if auto_reggressive:
        pred_answer = pred_answer.replace("answer: ", "")
        acc = 1 if answer in pred_answer else 0
    else:
        acc = 1 if pred_answer == answer else 0
    prob = np.exp(transition_scores[0].sum().cpu().numpy())

    data_to_csv.append([question, answer, pred_answer, acc, prob]) # , eo, ro

    # print("Question: ", question)
    # print("Answer: ", answer)
    # print("Predicted answer: ", pred_answer)
    # print("\n")

df = pd.DataFrame(data_to_csv, columns=["Question", "Answer", "Predicted Answer", "Accuracy", "Probability"]) # "Exact Ordering", "Relaxed Ordering"
val_ds_name = pargs.val.split("/")[-1].split(".")[0]
df.to_csv(f"./logs/{model_name}-finetuned2-{pargs.postfix}/results-{val_ds_name}.csv")
print("Saved to", f"./logs/{model_name}-finetuned2-{pargs.postfix}/results-{val_ds_name}.csv")