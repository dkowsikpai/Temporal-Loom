import spacy
import stanza
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re
import os 

class NER:

    def __init__(self, method="stanza", download_model=None, cuda="3") -> None:
        """
        method: spacy, stanza, transformer
        download_model: if method is stanza, download_model can be set to None or 
        """
        self.method = method
        print(method)

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda

        self.year_regex = re.compile(r'(\d{4})')
        if method == "spacy":
            self.nlp_spacy = spacy.load('en_core_web_sm') # https://spacy.io/models/en#en_core_web_sm
        elif method == "stanza":
            self.nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,ner', model_dir="./models/", download_method=download_model) # https://stanfordnlp.github.io/stanza/ner.html
        elif method == "transformer":

            # self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            # self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
            # self.nlp_transformer = pipeline("ner", model=model, tokenizer=tokenizer)

            self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
            self.model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
            self.nlp_transformer = pipeline("ner", model=self.model, tokenizer=self.tokenizer, grouped_entities=True)

    def _spacy_ner(self, text):
        doc = self.nlp_spacy(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def _year_object_pair(self, l):
        out = []
        for i in range(len(l)-1):
            if l[i][1] == 'DATE':
                year = l[i][0].replace('year', '')
                year = year.strip()
                out.append((year, l[i+1][0]))
        return out


    def _transformer_ner(self, text):
        return self.nlp_transformer(text)


    def _stanza_ner(self, text):
        doc = self.nlp_stanza(text)
        out = []
        for sent in doc.sentences:
            for ent in sent.ents:
                out.append((ent.text, ent.type))
        return out


    def get_year_object_pairs(self, text):
        if self.method == "transformer":
            years = self.year_regex.findall(text)
            print(years)
            objects = self._transformer_ner(text)
            out = []
            for i in range(len(years)):
                out.append((years[i], objects[i]["word"]))
            return out
        
        elif self.method == "stanza":
            return self._year_object_pair(self._stanza_ner(text))
        
        elif self.method == "spacy":
            return self._year_object_pair(self._spacy_ner(text))


if __name__ == '__main__':

    n = NER()

    l = ("In year 2010: S.V. Zulte Waregem. In year 2013: K.V. Kortrijk. In year 2016: Çaykur Rizespor.")
    print(l)
    print(n.get_year_object_pairs(l))


