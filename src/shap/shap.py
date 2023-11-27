from transformers import AutoModelForCausalLM, AutoTokenizer
# from matplotlib import pyplot as plt
import os 
import shap

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_checkpoint = "/home/dkowsik/temporal/results/gpt2-large-finetuned2/checkpoint-13500"
s = ["In year 2010: Donald Trump is a member of the "]

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to("cuda")

# set model decoder to true
model.config.is_decoder = True
# set text-generation params under task_specific_params
model.config.task_specific_params["text-generation"] = {
    "do_sample": True,
    "max_length": 50,
    "temperature": 0.7,
    "top_k": 50,
    "no_repeat_ngram_size": 2,
}

explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(s)


fig = shap.plots.text(shap_values, matplotlib=True, show=False)
plt.savefig("./results/shap.pdf")
