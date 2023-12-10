import pandas as pd
import os
import jsonlines

csvs = os.listdir('./data/worldometer/countries')
csvs = [csv for csv in csvs if csv.endswith('.csv')]

data = []
count = 0
total_changes = 0
for csv in csvs:
    country = csv.split('.')[0]
    country = ' '.join((country.split('-')[:-1]))
    country = country.casefold().title()
    df = pd.read_csv(f'./data/worldometer/countries/{csv}')
    d = {
        "query": f"Population of {country} is",
        "answer": [],
        "id": count,
    }
    for i, row in df.iterrows():
        d['answer'].append({
            "date": str(row['Year']),
            "answer": str(row['Population']),
        })
        total_changes += 1

    d["answer"].sort(key=lambda x: x['date'])

    data.append(d)
    count += 1

print(data[0])
print("Total countries:", len(data))
print("Total changes:", total_changes)

with jsonlines.open('./data/worldometer/worldometer.json', 'w') as f:
    f.write_all(data)
