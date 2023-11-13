from ner import NER
import pandas as pd
from tqdm import tqdm

get_year_object_pairs = NER(method="stanza").get_year_object_pairs

df = pd.read_csv("./data/ft-2010-2018.csv")
print(df.head())

data = []
for i in tqdm(range(len(df))):
    data.append(get_year_object_pairs(df["answer"][i]))

count = 0
for i, it in enumerate(data):
    flag = True
    if len(it) == 0:
        continue
    for y, o in it:
        if y not in df["answer"][i] or o not in df["answer"][i]:
            flag = False
            break
    if flag:
        count += 1

print(count)
print(len(data))  

with open("./data/test-ner.txt", "w") as f:
    for i in range(len(data)):
        f.write(df["answer"][i] + "\n")
        for y, o in data[i]:
            f.write(f"{y}\t{o}; ")
        f.write("\n")

# Spacy
# 16766
# 18298

# Stanza
# 17704
# 18298
