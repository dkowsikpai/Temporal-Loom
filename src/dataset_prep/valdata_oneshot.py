import argparse
import pandas as pd
import jsonlines

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

# Exists before but changed during this period
oneshot = []
for i in data:
    i["answer"].sort(key=lambda x: x["date"])
    if int(i["answer"][0]["date"]) < start:
        idx = 0
        while idx < len(i["answer"]) and int(i["answer"][idx]["date"]) < start:
            idx += 1
        idx -= 1
        if 0 <= idx < len(i["answer"]):
            oneshot.append({"question": "In year " + i["answer"][idx]["date"] + ": " + i["query"], "answer": i["answer"][idx]["answer"]})

print("Number of samples in oneshot:", len(oneshot))

df = pd.DataFrame(oneshot, columns=["question", "answer"])
year = args.dataset_path.split("/")[-1].split(".")[0]
df.to_csv(f"./data/oneshot-{year}-auto-gen.csv", index=False)