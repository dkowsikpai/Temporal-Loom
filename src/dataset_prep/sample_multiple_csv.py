# Importing necessary libraries
import jsonlines
from tqdm import tqdm
import argparse
from rich.pretty import pprint
import pandas as pd
import random

random.seed(42)

argparser = argparse.ArgumentParser()
argparser.add_argument('--input', help='Input file path')
argparser.add_argument('--sample', type=int, default=1000, help='Sample size')


args = argparser.parse_args()

df = pd.DataFrame(columns=["question", "answer"])
# Reading the input file
files = args.input.split(',')
for i in range(len(files)):
    print("Reading file {}".format(files[i]))
    temp = pd.read_csv(files[i])
    df = pd.concat([df, temp], ignore_index=True)
    

df = df.sample(n=args.sample, random_state=42)

# Writing the output file
print("Writing the output file")
df.to_csv('./data/sequential/combined_data.csv', index=False)
