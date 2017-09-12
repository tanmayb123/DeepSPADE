import numpy as np
import data_helpers
from w2v import train_word2vec
from sklearn.utils import class_weight
from keras.models import Model
from keras.layers import *
import keras

np.random.seed(2)

embedding_dim = 100
batch_size = 32
num_epochs = 1
val_split = 0.1

min_word_count = 1
context = 12

print("Loading data...")

x, y, vocabulary, vocabulary_inv = data_helpers.load_data()
embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context)

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices].argmax(axis=1)

print("Vocabulary Size: {:d}".format(len(vocabulary)))

layerinput = Input(shape=[267], dtype='int32')

encoded = Embedding(len(vocabulary), embedding_dim, input_length=267, weights=embedding_weights)(layerinput)
encoded = LSTM(100, return_sequences=True)(encoded)
#decoded = RepeatVector(267)(encoded)
decoded = LSTM(100, return_sequences=True)(encoded)

model = Model(inputs=[layerinput], outputs=[decoded])

model.compile(loss='binary_crossentropy', optimizer='adam')

model.summary()

model.fit(x_shuffled, x_shuffled, batch_size=batch_size, nb_epoch=num_epochs, validation_split=val_split, verbose=1)

model.save('autoencode.h5')
