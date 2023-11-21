import jsonlines
import argparse
import pandas as pd
import random

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--year', type=str, default="2010-2018")
parser.add_argument('--val-size', type=int, default=0)


args = parser.parse_args()
start_year = int(args.year.split("-")[0])
end_year = int(args.year.split("-")[1])

augmented_questions = {}
with jsonlines.open("./utils/templama_augmented_relations.jsonl") as reader:
    for line in reader:
        augmented_questions[line["relation"]] = line["paraphrase"]

with jsonlines.open(args.dataset_path) as reader:
    data = []
    val_data = []
    for line in reader:
        temp = {}
        # temp["question"] = line["query"]
        temp["subject"] = line["subject"]
        temp["relation"] = line["relation"]
        temp["all_answer"] = []
        for item in line["answer"]:
            if start_year <= int(item["date"]) <= end_year:
                temp["question"] = "In year " + item["date"] + ": " + line["query"]
                temp["answer"] = item["answer"] # "In year " + item["date"] + ": " + 
                temp["all_answer"].append(item)

                if "answer" in temp and random.random() < 0.4 and len(val_data) < args.val_size: #Randomly select 4000 for validation
                    val_data.append(temp.copy())

                elif "answer" in temp: # If not in that year, skip
                    data.append(temp.copy())

                del temp["answer"]

    print("Number of samples in train:", len(data))
    print("Number of samples in validation:", len(val_data))

    # aug_data = []
    # for i in data:
    #     for key, value in augmented_questions.items():
    #         if key in i["relation"]:
    #             for q in value:
    #                 if len(i["all_answer"]) > 0:
    #                     answer = ["In year " + a["date"] + ": " + a["answer"] for a in i["all_answer"] if a["answer"] != ""]
    #                     answer = ". ".join(answer) + '.'
    #                     aug_data.append({"question": q.replace("X", i["subject"]), "answer": answer})
    # data.extend(aug_data)

    # aug_data = []
    # for i in val_data:
    #     for key, value in augmented_questions.items():
    #         if key in i["relation"]:
    #             for q in value:
    #                 if len(i["all_answer"]) > 0:
    #                     answer = ["In year " + a["date"] + ": " + a["answer"] for a in i["all_answer"] if a["answer"] != ""]
    #                     answer = ". ".join(answer) + '.'
    #                     aug_data.append({"question": q.replace("X", i["subject"]), "answer": answer})
    # val_data.extend(aug_data)

    # delete all datafrom list
    for i in data:
        if "all_answer" in i:
            del i["all_answer"]
        if "subject" in i:
            del i["subject"]
        if "relation" in i:
            del i["relation"]
    for i in val_data:
        if "all_answer" in i:
            del i["all_answer"]
        if "subject" in i:
            del i["subject"]
        if "relation" in i:
            del i["relation"]

    random.shuffle(data)
    random.shuffle(val_data)

df = pd.DataFrame(data)
df.to_csv(f"./data/ft-{args.year}-auto-reg.csv", index=False)

if args.val_size > 0:
    df = pd.DataFrame(val_data)
    df.to_csv(f"./data/ft-val-{args.year}-auto-reg.csv", index=False)
