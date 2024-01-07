import pandas as pd
import argparse

from metrics import *

parser = argparse.ArgumentParser(description='Run metric on csv file')
parser.add_argument('--csv', type=str, help='csv file to run metric on')

args = parser.parse_args()

df = pd.read_csv(args.csv)

print(df.head())

