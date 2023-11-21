import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, required=True) # Train dataset in csv
parser.add_argument('--val-sample', type=int, default=1000)

args = parser.parse_args()

df = pd.read_csv(args.dataset_path)
df = df.sample(args.val_sample)

year = args.dataset_path.split("/")[-1].split(".")[0]
df.to_csv(f"./data/val-ft-{year}-auto-gen.csv", index=False)