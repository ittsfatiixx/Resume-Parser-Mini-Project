import pandas as pd

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

import json
f = open('train_Resume.json', encoding="utf8")
TRAIN_DATA = json.load(f)


for text, annot in tqdm(TRAIN_DATA['annotations']): 
    doc = nlp.make_doc(text) 
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents 
    db.add(doc)

db.to_disk("./resume_training_data.spacy") # save the docbin object

#  ! python -m spacy init config config.cfg --lang en --pipeline ner --optimize efficiency
#  ! python -m spacy train config.cfg --output ./ --paths.train ./resume_training_data.spacy --paths.dev ./resume_training_data.spacy