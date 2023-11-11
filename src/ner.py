import spacy

nlp = spacy.load('en_core_web_sm') # https://spacy.io/models/en#en_core_web_sm

def _ner(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def _year_object_pair(l):
    out = []
    for i in range(len(l)-1):
        if l[i][1] == 'DATE':
            year = l[i][0].replace('year', '')
            year = year.strip()
            out.append((year, l[i+1][0]))

    return out

def get_year_object_pairs(text):
    return _year_object_pair(_ner(text))


if __name__ == '__main__':
    l = _ner("In year 2010: S.V. Zulte Waregem. In year 2013: K.V. Kortrijk. In year 2016: Çaykur Rizespor.")
    print(l)
    print(_year_object_pair(l))


