import pandas as pd
from transformers import AutoTokenizer
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Run metric on csv file')
parser.add_argument('--csvs', type=str, help='csv file to run metric on')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

csvs = os.listdir(args.csvs)
csvs = [x for x in csvs if x.endswith(".csv")]
csvs = sorted(csvs)
csvs.remove("yearly_freq.csv")


lengths = []
for csv in tqdm(csvs, desc="CSVs"):
    df = pd.read_csv(os.path.join(args.csvs, csv))
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token = tokenizer.eos_token
    for i in tqdm(range(len(df)), desc="Rows", leave=False):
        prompt = df.iloc[i]["query"].strip() + " " + str(df.iloc[i]["answer"])
        lengths.append(len(tokenizer.tokenize(prompt)))


frequency_len = {}
for length in lengths:
    frequency_len[length] = frequency_len.get(length, 0) + 1

print("Total length:", len(lengths))
print("Total unique length:", len(frequency_len))
print("Average length:", sum(lengths) / len(lengths))
print("Max length:", max(lengths))
print("Min length:", min(lengths))

name = args.csvs.split("/")[-1]
plt.bar(frequency_len.keys(), frequency_len.values())
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.title("Length vs Frequency")
plt.savefig(f"./results/length_vs_frequency-{name}.png")

df = pd.DataFrame()
d = list(frequency_len.items())
d.sort(key=lambda x: x[0])
df["length"] = [x[0] for x in d]
df["frequency"] = [x[1] for x in d]
df.to_csv(f"./results/length_vs_frequency-{name}.csv", index=False)



