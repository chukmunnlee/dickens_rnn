import spacy
import numpy as np
import tensorflow as tf
import pickle
import re

from util import *
from tqdm import *

from tensorflow.keras.preprocessing.text import Tokenizer

#filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n—’‘“”'
filters='!"#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n—“”'
F = '!"#$%&\(\)\*\+,-\./:;<=>\?@\[\]\\\^_`\{\}|\~\t\n“”'

tokenizer = Tokenizer(num_words=200000, lower=True, char_level=False, filters=filters)

nlp = spacy.load('en_core_web_lg')

with open('./data/great_expectation.txt', 'r', encoding='utf-8-sig') as f:
   text_chunks = f.read().replace('\n', ' ').split('.')

LINES = 1000
LINES = len(text_chunks)

for i in tqdm(range(LINES)):
   #line = tokenize_line(text_chunks[i])
   line = text_chunks[i].strip().lower()
   line = re.sub(F, ' ', line)
   if len(line) <= 0:
      continue
   line = tokenize_line(line)
   tokenizer.fit_on_texts(line)
   seq = tokenizer.texts_to_sequences(line)
   txt = tokenizer.sequences_to_texts(seq)
   toks = nlp(' '.join(txt))
   for s, t in zip(seq, toks):
      print(t, t.pos, s[0])
   
   exit()
with open('./tokenizer.pickle', 'wb') as f:
   pickle.dump(tok, f)

print('word count: ', len(tok.word_counts))
print(tok.word_counts)
print('count = ', count)
