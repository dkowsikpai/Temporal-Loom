# A Temporal Loom for Models

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


## Execution of Code
### Combining Data

Use the following script
```term
python src/combine_restructure_data.py --input ./data/train.json,./data/test.json,./data/val.json
```
> Note: You can add more data by seperating the data path in `--input` parameter by comma.