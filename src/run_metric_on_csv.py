import pandas as pd
import argparse
from tqdm import tqdm

from metrics import *

parser = argparse.ArgumentParser(description='Run metric on csv file')
parser.add_argument('--csv', type=str, help='csv file to run metric on')
parser.add_argument('--num', type=str, help='numerical or non')

args = parser.parse_args()

print(args)

if args.num == "non":
    print("Running non-numerical metrics")
    metrics = {
        "exact_match": exact_match,
        "error": num_error,
        "ivf": ivf,
        "polarity": polarity,
    }
else:
    print("Running numerical metrics")
    metrics = {
        "accuracy": accuracy,
        "exact_ordering": exact_ordering,
        "relaxed_ordering": relaxed_ordering
    }


df = pd.read_csv(args.csv)
df = df.astype(str)

preds = []
for i in range(len(df)):
    preds.append(df.iloc[i]["prediction"].replace(df.iloc[i]["query"], ""))
df["prediction"] = preds
del preds

print(df.head())

for i in tqdm(range(len(df)), leave=False):
    for m in metrics.keys():
        val = metrics[m](df.iloc[i]["prediction"], df.iloc[i]["answer"])
        df.loc[i, m] = val
        # print(m, val)

df.to_csv("test.csv", index=False)

print("Mean metrics:")
for m in metrics.keys():
    print(m, df[m].mean())

