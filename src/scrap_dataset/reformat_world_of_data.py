import pandas as pd
import jsonlines

BASE_DIR = '/home/dkowsik/temporal/data/raw/nuclear-weapons/csvs/number-of-nuclear-weapons-tests.csv'
df = pd.read_csv(BASE_DIR)

num_to_class = {
    0: 'Does not consider',
    1: 'Considers',
    2: 'Pursues',
    3: 'Possesses',
    # 4: 'possessed',
    # 5: 'used',
    # 2: 'ratified',
}

rows = list(df.columns)[3:4]
print(rows)


for idx, r in enumerate(rows):
    data = {}
    for i, row in df.iterrows():
        if row['Entity'] not in data:
            data[row['Entity']] = []
        data[row['Entity']].append({
            'date': str(row['Year']),
            # 'answer': str(num_to_class[row['nuclear_weapons_status']]),
            'answer': str(row["nuclear_weapons_tests"]),
        })

    ndata = []
    for i, key in enumerate(data):
        data[key].sort(key=lambda x: x['date'])
        s = r.lower()
        # s = s.replace("Terrorism deaths", "Terrorism deaths by")
        ndata.append({
            "query": f"Number of nuclear weapon tests by {key} is",
            "answer": data[key],
            "id": i,
        })
        # data[key]["query"] = f"V-Dem Human rights index in {key} is"
        # data[key]["id"] = i


    file_name = BASE_DIR.split('/')[-1].split('.')[0]
    sub_folder = BASE_DIR.split('/')[-3]
    # st = r.find("(")
    # ed = r.find(")")
    # r = r[st+1: ed].replace("+", "")
    with jsonlines.open(f'/home/dkowsik/temporal/data/raw/{sub_folder}/{file_name}.json', 'w') as writer:
        writer.write_all(ndata)



    # print(data['Afghanistan'])
    # print(len(data['Afghanistan']))
    print(len(ndata))