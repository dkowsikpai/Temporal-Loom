import pandas as pd
import os
import jsonlines
from tqdm import tqdm
from rich.pretty import pprint

BASE_URL = "./data/raw/pollution"
CSV_URL = f"{BASE_URL}/"

csvs = os.listdir(CSV_URL)
csvs = [csv for csv in csvs if csv.endswith('.csv')]

data = []
count = 0
total_changes = 0
for csv in tqdm(csvs):
    # country = csv.split('.')[0]
    # country = ' '.join((country.split('-')[:-1]))
    # country = country.casefold().title()
    df = pd.read_csv(f'{CSV_URL}/{csv}')
    # print(df.columns)
    # if df.shape[1] != 7:
    #     continue
    # df.columns = ['Year', 'GDP Nominal  (Current USD)', 'GDP Real (Inflation adj.)',
    #    'GDP change', 'GDP per capita', 'Pop. change', 'Population']
    # print(df.head())
    # exit()
    
    for i, row in df.iterrows():
        d = {
            "query": f"CO2 level in {row['Country Name']} in metric tons per capita is",
            # "query": f"Population change in world is",
            "answer": [],
            "id": count,
        }

        for y, r in enumerate(row.to_list()[10:]):
            if len(str(r)) == 0 or str(r) == 'nan':
                continue
            d['answer'].append({
            "date": str(1960+y),
            "answer": str(r),
        })
        # exit()
        # d['answer'].append({
        #     "date": str(row['Year']),
        #     "answer": str(row['Pop. change']),
        # })
        total_changes += 1

        d["answer"].sort(key=lambda x: x['date'])

        if len(d['answer']) == 0:
            continue
        data.append(d)
        count += 1

print(data[0])
print("Total countries:", len(data))
print("Total changes:", total_changes)

with jsonlines.open(f'{BASE_URL}/climate_change.json', 'w') as f:
    f.write_all(data)
