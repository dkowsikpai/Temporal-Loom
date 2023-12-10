# A Temporal Loom for Models
_We see the timeline as physical dimension, of course it is in DL model training LOL!_

## Installation
Requirements
- conda environment (conda>=4.12.0)
- python>=3.8

Run the installation script 
```term
bash install.sh
```

## Dataset
Create a folder named `data` after cloning this repository and download the [TempLAMA](https://github.com/google-research/language/tree/master/language/templama) dataset.
There are a total of 50,310 samples after the combing of the data. 

### Analysis
Number of datasamples that have changed over time (unique queries) = 5,823

More are available in the [Google Colab](https://colab.research.google.com/drive/1Rwz7tQKBNxo8l-gSoP8dX21rWBXLNeRn?usp=sharing)


## Execution of Code
### Combining Data

Use the following script
1. First command converts the dataset to the dataset of diff format (Contains only the change where the subject changed over the year)
2. Second command converts the dataset to the csv format - question, answer format
```term
python src/dataset_prep/combine_restructure_data.py --input ./data/train.json,./data/test.json,./data/val.json
python src/dataset_prep/finetuning_data_jsonl_to_csv.py --dataset-path ./data/restructured_data.json --year 2010-2018
```
> Note: You can add more data by seperating the data path in `--input` parameter by comma.

### Paraphrasing of the input query (relation)
The list of the paraphrased relation can be seen in the file `utils/templama_relation_rephrase.jsonl` for the TempLAMA dataset.


### Finetuning dataset
Number of samples in train: 9149
Number of samples in validation: 3000

For **zeroshot** there are two cases,
1. Not at all seen dataset, Newly added information in that particular year
2. Previously seen but the year in the query is changed
```term
python src/dataset_prep/valdata_zeroshot.py --dataset-path ./data/restructured_data.json --val-year 2019-2020
```

For **oneshot** there  is one case
1. Sampled from the finetuning dataset - 1000 samples
```term
python src/dataset_prep/valdata_oneshot.py --dataset-path ./data/ft-2010-2018.csv --val-sample 1000
```


1. For T5-model: We replace `_X_` mask provided in the dataset with `<extra_id_0>` which is by default mask for the T5 model.
2. For GPT2 model: 

Finetune the model using the code
```term
python src/seq2seq.py --model t5-base --train ./data/ft-2010-2018.csv --val ./data/ft-val-2010-2018.csv --cuda 3
```

#### Sequential Training
```term
python src/seq2seq.py --train ./data/sequential/ft-2010-2015-auto-reg.csv --val ./data/sequential/ft-val-2010-2015-auto-reg.csv --cuda 3 --model t5-large | tee logs/t5-large-auto.txt

python src/seq2seq.py --train ./data/sequential/ft-2010-2015-auto-reg.csv --val ./data/sequential/ft-val-2010-2015-auto-reg.csv --cuda 3 --model gpt2-large | tee logs/gpt2-large-auto.txt
```


### Temsorboard
```term
python -m tensorboard.main --logdir ./logs --port 9000
python -m tensorboard.main --logdir ./results --port 8000
```

### Test
#### Generate output
```term
python src/test/generate_txt.py --cuda 3 --test-model-path python src/test/generate_txt.py --cuda 3 --test-model-path /home/dkowsik/temporal/results/gpt2-large-finetuned2/checkpoint-13500
```