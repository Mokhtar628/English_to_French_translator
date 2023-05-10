import numpy as np
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer

print("Load data")

with open("./NLP_projectDataset/en.txt") as file:
    en_data = ["sos "+line.rstrip()+" eos" for line in file]
with open("./NLP_projectDataset/fr.txt") as file:
    fr_data = ["sos " + line.rstrip() +" eos" for line in file]

print("Tockenization started")

# Define a Keras Tokenizer
en_tok = Tokenizer(num_words=230, oov_token='UNK')
en_tok.fit_on_texts(en_data)
fr_tok= Tokenizer(num_words=358, oov_token='UNK')
fr_tok.fit_on_texts(fr_data)

print("Tockenization compeleted")

def en_sents2seqs(input_type, sentences, onehot=False, pad_type='post', reverse=False):
    encoded_text = en_tok.texts_to_sequences(sentences)
    preproc_text = pad_sequences(encoded_text, padding=pad_type, truncating='post', maxlen=15)
    if reverse:
      # Reverse the text using numpy axis reversing
      preproc_text = preproc_text[:,::-1]
    if onehot:
        preproc_text = to_categorical(preproc_text, num_classes=230)
    return preproc_text



en_sent = ["sos new jersey is sometimes quiet during autumn eos"]
en_seq = en_sents2seqs('source', en_sent, onehot=True, reverse=False)
print(en_seq)

nmt = load_model('./NLP_projectDataset/')

fr_pred = nmt.predict(en_seq)
fr_seq = np.argmax(fr_pred, axis=-1)[0]

print(fr_seq)
translation = ''
for i in fr_seq:
  if i == 0:break
  translation += ' ' + fr_tok.index_word[i]