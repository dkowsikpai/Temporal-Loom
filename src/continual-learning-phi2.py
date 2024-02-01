import os 
import time

start_year = 1947
end_year = 2022
dataset_path = "/home/dkowsik/temporal-dataset/data/micro_dataset/final_top100_jsons_shrunk_acc_prompt_yearly_prompt"
model_name = "microsoft/phi-2"
prev = None
seed = 1
batch_size = 1
numerical = "numerical"
epoch = 10
learning_rate = 3e-5
save_limit = 1
patience = 4
prefix = "Genereate only the number for the following query: "
cuda = "0"
logs_dir = "./logs/continual_phi2-micro"

os.makedirs(logs_dir, exist_ok=True)
for year in range(start_year, end_year+1):
    print(f"Year: {year}")
    if year != start_year and prev != None:
        model_name = prev

    command = f"CUDA_VISIBLE_DEVICES={cuda} python src/seq2seq-phi2 --dataset-path {dataset_path} --start-year {year} --end-year {year} --model-name {model_name} --seed {seed} --batch-size {batch_size} --num {numerical} --epochs {epoch} --lr {learning_rate} --save-limit {save_limit} --patience {patience} --prefix \"{prefix}\" | tee {logs_dir}/{year}.txt"
    print(command)
    os.system(command)

    # Set prev model
    with open(f"{logs_dir}/{year}.txt", "r") as f:
        for line in f.readlines():
            if line.find("Saving to") != -1:
                prev_chkpt = line.split("Saving to ")[1].split("\n")[0]
                dirs = os.listdir(prev_chkpt)
                if len(dirs) == 0:
                    print("No checkpoint found")
                    exit()
                dirs.sort(reverse=True) # The last checkpoint is the best one
                for d in dirs:
                    if d.find("checkpoint") != -1:
                        prev = f"{prev_chkpt}/{d}"
                        break
    print("="*100)
    time.sleep(5)