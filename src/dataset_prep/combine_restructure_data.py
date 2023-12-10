# Importing necessary libraries
import jsonlines
from tqdm import tqdm
import argparse
from rich.pretty import pprint
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument('--input', help='Input file path')

args = argparser.parse_args()

exclude_relations = ["is the head coach of", "is the chair of", "is the head of the government of"]

# Read the paraphrased relations from the file
print("Reading the paraphrased relations from the file")
with jsonlines.open('./utils/templama_relation_rephrase.jsonl') as f:
      para_relations = list(f)

# pprint(relations)

# Number of files to combine
print("Combining {} files".format(len(args.input.split(','))))

# Reading the input file
files = args.input.split(',')
for i in range(len(files)):
    print("Reading file {}".format(files[i]))
    with jsonlines.open(files[i]) as f:
        if i == 0:
            output = list(f)
        else:
            output.extend(list(f))

# Writing the output file
print("Writing the output file")
with jsonlines.open('./data/combined_data.json', mode='w') as writer:
    writer.write_all(output)


# Sample data
"""{
   "query":"Valentino Rossi plays for _X_.",
   "answer":[
      {
         "wikidata_id":"Q1085474",
         "name":"Yamaha Motor Racing"
      }
   ],
   "date":"2010",
   "id":"Q169814_P54_2010",
   "most_frequent_answer":{
      "wikidata_id":"Q1085474",
      "name":"Yamaha Motor Racing"
   },
   "most_recent_answer":{
      "wikidata_id":"Q1085474",
      "name":"Yamaha Motor Racing"
   },
   "relation":"P54"
}"""


# Restructuring the data
print("Restructuring the data")
grouped_data = {} # Keys will be the query and values will be dict of answers over time
for i in tqdm(output):
  i["query"] = i["query"].replace(".", "")
  if i["query"] not in grouped_data:
    grouped_data[i["query"]] = []
  grouped_data[i["query"]].append({"answer": i["answer"][0]["name"], "date": i["date"]})

# Remove recurring answers over the time (Keepining oly the change over the year)
print("Removing recurring answers over the time")
for k, v in grouped_data.items():
   filtered = []
   prev = ""
   for i in v:
      if i["answer"] != prev:
         filtered.append(i)
         prev = i["answer"]
   grouped_data[k] = filtered

pprint(grouped_data["Valentino Rossi plays for _X_"])

# Removing repetitive duplicates
print("Number of samples before filtering: {}".format(len(grouped_data)))
filtered = {}
for k, v in grouped_data.items():
  if len(v) < 12:
    filtered[k] = v
grouped_data = filtered.copy()
print("Number of samples after filtering: {}".format(len(grouped_data)))

# Subject relation extraction
print("Extracting subject relation")
df = pd.read_csv('./data/templates.csv')

template_relation = df["Template"].to_list()
relations = []
for i in template_relation:
    rel = i.replace("<subject>", "")
    rel = rel.replace("<object>", "")
    rel = rel.replace(".", "")
    rel = rel.strip()
    if rel not in exclude_relations:
        relations.append(rel)

print("Relations in the dataset are:")
pprint(relations)
print("Number of relations", len(relations))

def exclude(query):
   for rel in exclude_relations:
      if rel in query:
         return True
   return False

# JSONing
data = []
for k, v in grouped_data.items():
    if not exclude(k):
        data.append({"query": k, "answer": v})


relations_dist = {}
for i in range(len(data)):
   for rel in relations:
      if rel in data[i]["query"]:
         data[i]["relation"] = rel
         subject = data[i]["query"].replace(rel, "")
         subject = subject.replace("_X_", "")
         subject = subject.replace(".", "")
         subject = subject.strip()
         data[i]["subject"] = subject

         # # Add Paraphrased sentences
         # data[i]["paraphrased_query"] = []
         # for rel_phrase in para_relations:
         #    if rel_phrase["relation"] == rel:
         #       paraphrases = []
         #       for para_phase in rel_phrase["paraphrase"]:
         #          if rel_phrase["subject_first"]:
         #             para = subject + " " + para_phase + " _X_" 
         #          else:
         #             para = "_X_ " + para_phase + " " + subject + "."
         #          paraphrases.append(para)
         #       data[i]["paraphrased_query"] = paraphrases

# Give id for each sample
count = 1
for i in range(len(data)):
   data[i]["id"] = count
   count += 1

# Writing the output file
print("Writing the output file")
with jsonlines.open('./data/sequential/restructured_data_auto_gen.json', mode='w') as writer:
    writer.write_all(data)

