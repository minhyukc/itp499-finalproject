# Brian Choi
# ITP 499 Fall 2022
# Final Project
# Problem 2

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, Dropout
import matplotlib.pyplot as plt

# 1.	Write code to train an RNN model that translates English into Spanish.
# The dataset is provided to you as a csv file consisting of sentence pairs.
df = pd.read_csv('/content/drive/MyDrive/english-spanish-dataset.csv')

df.drop(df.columns[[0]], axis=1, inplace=True)

# 2. Process the dataset into English sentences and Spanish sentences.
# To reduce the size of the corpus, you can limit to the first 50,000 sentences.
english = df.iloc[:50000,0]
spanish = df.iloc[:50000,1]

# 3. Tokenize the setences in both languages
def tokenize(x):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(x)
  return tokenizer.texts_to_sequences(x), tokenizer

# 4. Pad the sentences as necessary
def pad(x, length=None):
  return pad_sequences(x, maxlen=length, padding='post')

def preprocess (x, y):
  preprocess_x, x_tk = tokenize(x)
  preprocess_y, y_tk = tokenize(y)
  preprocess_x = pad(preprocess_x)
  preprocess_y = pad(preprocess_y)
  preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
  return preprocess_x, preprocess_y, x_tk, y_tk

preproc_eng_sentences, preproc_esp_sentences, eng_tokenizer, esp_tokenizer = preprocess(english, spanish)

max_eng_sequence_len = preproc_eng_sentences.shape[1] # 8
max_esp_sequence_len = preproc_esp_sentences.shape[1] # 12
eng_size = len(eng_tokenizer.word_index) # 6907
esp_size = len(esp_tokenizer.word_index) # 12959

def logits_to_text (logits, tokenizer):
  index_to_words = {id:word for word, id in tokenizer.word_index.items()}
  index_to_words[0] = "<PAD>"
  return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

# 5. Build an RNN model with the following layers
    # a. At least 1 GRU layer
    # b. At least 1 dropout layer
    # c. At least 1 dense layer
# 6. Use the sparse categorical crossentropy loss function
def rnn (input_shape, output_sequence_len, eng_size, esp_size):
  model = Sequential()
  model.add(GRU(128, input_shape=input_shape[1:], return_sequences=True))
  model.add(Dropout(0.5))
  model.add(GRU(128, return_sequences=True))
  model.add(Dropout(0.5))
  model.add(TimeDistributed(Dense(256, activation='relu')))
  model.add(Dropout(0.5))
  model.add(TimeDistributed(Dense(esp_size + 1, activation='softmax')))
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

tmp_x = pad(preproc_eng_sentences, max_esp_sequence_len)
tmp_x = tmp_x.reshape((-1, preproc_esp_sentences.shape[-2], 1))

model = rnn(
    tmp_x.shape,
    max_esp_sequence_len,
    eng_size,
    esp_size
)

# 7. Train the model for at least 5 epochs
h = model.fit(tmp_x, preproc_esp_sentences, batch_size=300, epochs=5, validation_split=0.2)

# 8. Plot the loss and accuracy curves for the train and validation sets (screenshot)
# Loss Curve
plt.figure(figsize=[6,4])
plt.plot(h.history['loss'], 'black', linewidth=2.0)
plt.plot(h.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)
plt.show()

# Accuracy Curve
plt.figure(figsize=[6,4])
plt.plot(h.history['accuracy'], 'black', linewidth=2.0)
plt.plot(h.history['val_accuracy'], 'green', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)
plt.show()

# 9. Prompt the user to enter an English sentence.
# this german hotel belongs to the company
# Translate to Spanish.
user_input = input('Enter English sentence: ')
print('You entered:', user_input)
user_input = [eng_tokenizer.word_index[word] for word in user_input.split()]
user_input = pad_sequences([user_input], maxlen=preproc_esp_sentences.shape[-2], padding='post')
tmp_x = user_input.reshape((-1, preproc_esp_sentences.shape[-2], 1))
prediction = model.predict(tmp_x)
print("Translation is", logits_to_text(prediction[0], esp_tokenizer))
