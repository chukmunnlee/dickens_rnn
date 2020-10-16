import pickle
import spacy
import numpy as np
import tensorflow as tf

from util import *
from tqdm import *

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=200000, lower=True,
      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n—’‘“”')

nlp = spacy.load('en_core_web_lg')

with open('./data/great_expectation.txt', 'r', encoding='utf-8-sig') as f:
   text_chunks = f.read().replace('\n', ' ').split('.')

LINES = 1000
LINES = len(text_chunks)

for i in tqdm(range(LINES)):
   #line = tokenize_line(text_chunks[i])
   line = text_chunks[i]
   if len(line) <= 0:
      continue
   tokenizer.fit_on_texts(line)
   seq = tokenizer.texts_to_sequences([ line ])
   tok = tokenizer.sequences_to_texts(seq)

   print('seq = ', seq)
   print('tok = ', tok)
   exit()


with open('./tokenizer.pickle', 'wb') as f:
   pickle.dump(tok, f)

print('word count: ', len(tok.word_counts))
print(tok.word_counts)
print('count = ', count)
