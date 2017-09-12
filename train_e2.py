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

layers1 = Embedding(len(vocabulary), embedding_dim, input_length=267, weights=embedding_weights)(layerinput)
layers1 = Conv1D(128, 3, padding='valid', activation='relu', strides=1)(layers1)
layers1 = Dropout(0.5)(layers1)
layers1 = MaxPooling1D(pool_size=3)(layers1)
layers1 = Conv1D(128, 4, padding='valid', activation='relu', strides=1)(layers1)
layers1 = Dropout(0.5)(layers1)
layers1 = MaxPooling1D(pool_size=3)(layers1)
layers1 = Conv1D(128, 4, padding='valid', activation='relu', strides=1)(layers1)
layers1 = Dropout(0.5)(layers1)
layers1 = MaxPooling1D(pool_size=4)(layers1)

layers2 = Conv1D(128, 4, padding='valid', activation='relu', strides=1)(layers1)
layers2 = Flatten()(layers2)
layers2 = Dropout(0.5)(layers2)
layers2 = Dense(40, activation='sigmoid')(layers2)

layers2_1 = Conv1D(128, 3, padding='valid', activation='relu', strides=1)(layers1)
layers2_1 = Dropout(0.5)(layers2_1)
layers2_1 = GRU(20, return_sequences=False)(layers2_1)

layers3 = GRU(80, return_sequences=True)(layers1)
layers3 = Dropout(0.5)(layers3)
layers3 = GRU(40, return_sequences=True)(layers3)
layers3 = Dropout(0.5)(layers3)
layers3 = GRU(20, return_sequences=False)(layers3)
#layers3 = Dropout(0.5)(layers3)
#layers3 = Flatten()(layers3)

layers4 = keras.layers.concatenate([layers2, layers2_1, layers3], axis=-1)
layers4 = Dropout(0.5)(layers4)
layers4 = Dense(30, activation='sigmoid')(layers4)
layers4 = Dropout(0.5)(layers4)
layers4 = Dense(1, activation='sigmoid')(layers4)

model = Model(inputs=[layerinput], outputs=[layers4])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

#model.load_weights('train_ensemble_try3_1.h5')

model.fit(x_shuffled, y_shuffled, batch_size=batch_size, nb_epoch=num_epochs, validation_split=val_split, verbose=1, class_weight=class_weight.compute_class_weight('balanced', np.unique(y_shuffled), y_shuffled))

model.save('train_ensemble_try3_1.h5')
