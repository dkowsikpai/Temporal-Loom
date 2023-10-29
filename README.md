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
```term
python src/combine_restructure_data.py --input ./data/train.json,./data/test.json,./data/val.json
```
> Note: You can add more data by seperating the data path in `--input` parameter by comma.

### Paraphrasing of the input query (relation)
The list of the paraphrased relation can be seen in the file `utils/templama_relation_rephrase.jsonl` for the TempLAMA dataset.
