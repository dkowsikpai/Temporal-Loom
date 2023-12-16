import pandas as pd
import jsonlines

df = pd.read_csv('./data/raw/human-rights-index/human-rights-index-vdem.csv')

data = {}

for i, row in df.iterrows():
    if row['Entity'] not in data:
        data[row['Entity']] = []
    data[row['Entity']].append({
        'date': str(row['Year']),
        'answer': str(row['civ_libs_vdem_owid']),
    })

ndata = []
for i, key in enumerate(data):
    data[key].sort(key=lambda x: x['date'])
    ndata.append({
        "query": f"V-Dem Human rights index in {key} is",
        "answer": data[key],
        "id": i,
    })
    # data[key]["query"] = f"V-Dem Human rights index in {key} is"
    # data[key]["id"] = i


with jsonlines.open('./data/raw/human-rights-index/human-rights-index-vdem.json', 'w') as writer:
    writer.write_all(ndata)

# print(data['Afghanistan'])
# print(len(data['Afghanistan']))
# print(len(data))