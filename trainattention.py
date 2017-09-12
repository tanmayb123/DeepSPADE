"""
Train convolutional network for sentiment analysis. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For 'CNN-non-static' gets to 82.1% after 61 epochs with following settings:
embedding_dim = 20
filter_sizes = (3, 4)
num_filters = 3
dropout_prob = (0.7, 0.8)
hidden_dims = 100

For 'CNN-rand' gets to 78-79% after 7-8 epochs with following settings:
embedding_dim = 20
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

For 'CNN-static' gets to 75.4% after 7 epochs with following settings:
embedding_dim = 100
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

* it turns out that such a small data set as "Movie reviews with one
sentence per review"  (Pang and Lee, 2005) requires much smaller network
than the one introduced in the original article:
- embedding dimension is only 20 (instead of 300; 'CNN-static' still requires ~100)
- 2 filter sizes (instead of 3)
- higher dropout probabilities and
- 3 filters per filter size is enough for 'CNN-non-static' (instead of 100)
- embedding initialization does not require prebuilt Google Word2Vec data.
Training Word2Vec on the same "Movie reviews" data set is enough to
achieve performance reported in the article (81.6%)

** Another distinct difference is slidind MaxPooling window of length=2
instead of MaxPooling over whole feature map as in the article
"""

import numpy as np
import data_helpers
from w2v import train_word2vec
from sklearn.utils import class_weight
#import tensorflow as tf
#sess = tf.Session()

from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
from keras import callbacks
#K.set_session(sess)

np.random.seed(2)

# Parameters
# ==================================================
#
# Model Variations. See Kim Yoon's Convolutional Neural Networks for
# Sentence Classification, Section 3 for detail.

model_variation = 'CNN-non-static'  #  CNN-rand | CNN-non-static | CNN-static
print('Model variation is %s' % model_variation)

# Model Hyperparameters
sequence_length = 45
embedding_dim = 20
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
context = 10        # Context window size

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

# Building model
# ==================================================
#
# graph subnet with one input and one output,
# convolutional layers concateneted in parallel
graph_in = Input(shape=(sequence_length, embedding_dim))
convs = []
for fsz in filter_sizes:
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=fsz,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)

if len(filter_sizes)>1:
    out = Merge(mode='concat')(convs)
else:
    out = convs[0]

graph = Model(input=graph_in, output=out)

#----------------------------------------------------------

remote = callbacks.RemoteMonitor(root='http://localhost:9000')


units = 100

embedded = Embedding(len(vocabulary), embedding_dim, input_length=247, weights=embedding_weights, trainable=True)

sequence_input = Input(shape=[247], dtype='int32')
embedded_sequences = embedded(sequence_input)

activations = LSTM(units, return_sequences=True)(embedded_sequences)
activations = LSTM(units, return_sequences=True)(activations)
activations = Dropout(0.4)(activations)
activations = LSTM(30, return_sequences=True)(activations)

# compute importance for each step
attention = Dense(1, activation='tanh')(activations)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(30)(attention)
attention = Permute([2, 1])(attention)


sent_representation = merge([activations, attention], mode='mul')
sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(30,))(sent_representation)

probabilities = Dense(1, activation='sigmoid')(sent_representation)

model = Model(input=sequence_input, output=probabilities)

#----------------------------------------------------------

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Training model
# ==================================================

model.fit(x_shuffled, y_shuffled, batch_size=batch_size, nb_epoch=num_epochs, validation_split=val_split, verbose=1, class_weight=class_weight.compute_class_weight('balanced', np.unique(y_shuffled), y_shuffled), callbacks=[remote])

model.save('save_tmp.h5')

#model.load_weights('save_tmp.h5')
#print model.evaluate(x_shuffled, y_shuffled)

#print model.predict(x_shuffled)
#print y_shuffled
#print len(y_shuffled)
