import spacy, pickle
import numpy as np
import tensorflow as tf

from tqdm import *

from util import *

from tensorflow.keras.preprocessing.text import Tokenizer

nlp = spacy.load('en_core_web_lg')

with open('./data/great_expectation.txt', 'r', encoding='utf-8-sig') as f:
   text_chunks = f.read().replace('\n', ' ').split('.')

with open('./tokenizer.pickle', 'rb') as f:
   tokenizer = pickle.load(f)

INDEX=10

seq = tokenizer.texts_to_sequences(text_chunks[INDEX])
toks = [ w for w in nlp(text_chunks[INDEX]) ]

zz = [ (t.text, t.pos_, s) for t, s in zip(toks, seq) ]
print(zz)

