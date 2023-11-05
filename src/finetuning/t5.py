from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import jsonlines
import random
from rich.pretty import pprint
import os
import json
from tqdm import tqdm
import difflib
import evaluate
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--year', type=str, required=True, default="2010")
parser.add_argument('--cuda', type=str, default=0)
args = parser.parse_args()


############################# Configurations #############################
# Set all seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)

# os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

# Hyperparameters
batch_size = 4
num_epochs = 10
learning_rate = 2e-5
max_length = 120

# Device
device = torch.device(f"cuda:3" if torch.cuda.is_available() else "cpu")

bleu = evaluate.load("bleu")

augmented_questions = {}
with jsonlines.open("./utils/templama_augmented_relations.jsonl") as reader:
    for line in reader:
        augmented_questions[line["relation"]] = line["paraphrase"]

################################################################################

# Example dataset - Replace this with your own dataset
class QADataset(Dataset):
    def __init__(self, dataset_path):
        with jsonlines.open(dataset_path) as reader:
            self.data = []
            for line in reader:
                temp = {}
                temp["question"] = line["query"]
                temp["subject"] = line["subject"]
                temp["relation"] = line["relation"]
                for item in line["answer"]:
                    if item["date"] == args.year:
                        temp["answer"] = item["answer"] + " in year " + args.year

                if "answer" in temp: # If not in that year, skip
                    self.data.append(temp)

        # Augmented questions
        aug_data = []
        for i in self.data:
            for key, value in augmented_questions.items():
                if key in i["relation"]:
                    for q in value:
                        aug_data.append({"question": q.replace("X", i["subject"]), "answer": i["answer"]})

        self.data += aug_data

        # Shuffle the data
        random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["question"], self.data[idx]["answer"]

def ids_to_string(ids):
    return tokenizer.decode(ids, skip_special_tokens=True)

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    b_score = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_questions, batch_answers = batch
            inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True)
            labels = tokenizer(batch_answers, return_tensors="pt", padding=True, truncation=True)

            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            labels = labels.input_ids.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_ids)
            predicted_token_ids = torch.argmax(outputs.logits, dim=-1)

            for i in range(len(predicted_token_ids)):
                predicted_answer = tokenizer.decode(predicted_token_ids[i], skip_special_tokens=True)
                actual_answer = batch_answers[i]
                total_predictions += 1
                ratio = difflib.SequenceMatcher(None, predicted_answer.strip().lower(), actual_answer.strip().lower()).ratio()
                correct_predictions += ratio

                # Compute BLEU score
                try:
                    b_score += bleu.compute(predictions=[predicted_answer.strip().lower()], references=[actual_answer.strip().lower()])["bleu"]
                except ZeroDivisionError:
                    pass

    accuracy = correct_predictions / total_predictions
    b_score = b_score / total_predictions
    return {"accuracy": accuracy, "bleu": b_score}


# Initialize the GPT-2 tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')
model.to(device)

# Prepare the dataset and dataloader
dataset = QADataset(args.dataset_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Print 10 samples of dataset
# for i in range(10):
#     pprint(dataset[i])

# exit()

# Fine-tuning loop
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

history = {"loss": [], "accuracy": [], "bleu": []}
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(dataloader):
        batch_questions, batch_answers = batch
        inputs = tokenizer(batch_questions, return_tensors="pt",  truncation=True, max_length=max_length, padding="max_length")
        labels = tokenizer(batch_answers, return_tensors="pt",  truncation=True, max_length=max_length, padding="max_length")

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        labels = labels.input_ids.to(device)

        # print(input_ids)
        # print(labels)
        # exit()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item()}")
        history["loss"].append(loss.item())

    test_accuracy = evaluate(model, dataloader)
    print(f"Test accuracy: {test_accuracy}")
    history["accuracy"].append(test_accuracy["accuracy"])
    history["bleu"].append(test_accuracy["bleu"])
    # Print average loss
    avg = sum(history["loss"]) / len(history["loss"])
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg}")


    # print sample predictions
    model.eval()
    for i in range(10):
        inputs = tokenizer(dataset[i][0], return_tensors="pt",  truncation=True, max_length=max_length, padding="max_length")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        print(f"Question: {dataset[i][0]}")
        print(f"Answer: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        print(f"Actual answer: {dataset[i][1]}")
        print()


# Save the fine-tuned model
ct = datetime.datetime.now()
ct = ct.strftime("%Y-%m-%d-%H-%M-%S")
model.save_pretrained("./results/fine_tuned_t5_base_model")
with open(f"./results/history_t5_base_{ct}.json", "w") as f:
    json.dump(history, f, indent=4)
