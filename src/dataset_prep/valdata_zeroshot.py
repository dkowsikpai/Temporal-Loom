import argparse
import jsonlines
import pandas as pd
import random

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, required=True) # Restructured dataset
parser.add_argument('--val-year', type=str, default="2019-2020")

args = parser.parse_args()

start = int(args.val_year.split("-")[0])
end = int(args.val_year.split("-")[1])

previouse_year = int(start) - 1

data = []
with jsonlines.open(args.dataset_path) as reader:
    for line in reader:
        data.append(line)

zeroshot = []

zeroshot_year_change = []
# previously_seen = set()

for i in data:
    i["answer"].sort(key=lambda x: x["date"])
    if int(i["answer"][0]["date"]) < start:
        idx = 0
        while idx < len(i["answer"]) and int(i["answer"][idx]["date"]) < start: # Take the latest answer before start year
            idx += 1
        idx -= 1
        if 0 <= idx < len(i["answer"]):
            zeroshot_year_change.append({"question": "In year " + str(start) + ": " + i["query"], "answer": i["answer"][idx]["answer"]})

        if "Carlos Sainz Jr plays for _X_" in i["query"]:
            print(i["answer"], idx)
        continue

    # Only consider data that was not in training set
    for a in i["answer"]:
        if start <= int(a["date"]) <= end:
            zeroshot.append({"question": "In year " + a["date"] + ": " + i["query"], "answer": a["answer"]})

print("Number of samples in zeroshot:", len(zeroshot))
print("Number of samples in zeroshot year change:", len(zeroshot_year_change))

random.shuffle(zeroshot)
random.shuffle(zeroshot_year_change)

df = pd.DataFrame(zeroshot, columns=["question", "answer"])
df.to_csv(f"./data/zeroshot-{args.val_year}-auto-gen.csv", index=False)

df = pd.DataFrame(zeroshot_year_change, columns=["question", "answer"])
df.to_csv(f"./data/zeroshot-year-change-{args.val_year}-auto-gen.csv", index=False)

