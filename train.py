import numpy as np
import data_helpers
from w2v import train_word2vec
from sklearn.utils import class_weight
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K

np.random.seed(2)

model_variation = 'CNN-non-static'  #  CNN-rand | CNN-non-static | CNN-static
print('Model variation is %s' % model_variation)

# Model Hyperparameters
sequence_length = 45
embedding_dim = 80
filter_sizes = (3, 4)
num_filters = 128
dropout_prob = (0.25, 0.5)
hidden_dims = 128

# Training parameters
batch_size = 32
num_epochs = 1
val_split = 0.1

# Word2Vec parameters, see train_word2vec
min_word_count = 1  # Minimum word count
context = 12        # Context window size

# Data Preparatopn
# ==================================================
#
# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data()

if model_variation=='CNN-non-static' or model_variation=='CNN-static':
    embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context)
    if model_variation=='CNN-static':
        x = embedding_weights[0][x]
elif model_variation=='CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')

# Shuffle data
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices].argmax(axis=1)

print("Vocabulary Size: {:d}".format(len(vocabulary)))

# main sequential model
model = Sequential()
if not model_variation=='CNN-static':
    model.add(Embedding(len(vocabulary), embedding_dim, input_length=267,
                        weights=embedding_weights))

model.add(Conv1D(128, 3, padding='valid', activation='relu', strides=1))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(128, 4, padding='valid', activation='relu', strides=1))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(128, 4, padding='valid', activation='relu', strides=1))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=4))
model.add(GRU(80, return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(40, return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(20))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.load_weights('save_GRU_80_1.h5')

# Training model
# ==================================================

model.fit(x_shuffled, y_shuffled, batch_size=batch_size, nb_epoch=num_epochs, validation_split=val_split, verbose=1, class_weight=class_weight.compute_class_weight('balanced', np.unique(y_shuffled), y_shuffled))
# class_weight=class_weight.compute_class_weight('balanced', np.unique(y_shuffled), y_shuffled))

model.save('save_GRU_80_2.h5')

#model.load_weights('save_tmp.h5')
#print model.evaluate(x_shuffled, y_shuffled)

#print model.predict(x_shuffled)
#print y_shuffled
#print len(y_shuffled)
