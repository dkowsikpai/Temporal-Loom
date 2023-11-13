import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True)
parser.add_argument('--xlsx', type=str, required=True)

pargs = parser.parse_args()

reviews = pd.read_csv(pargs.csv)
try:
    reviews = reviews.drop(columns=['Unnamed: 0'])
except:
    pass
reviews.to_excel(pargs.xlsx, index=False)
