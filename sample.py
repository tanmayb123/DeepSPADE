# -*- coding: utf-8 -*-
import data_helpers
import keras
from keras.models import Model
from keras.layers import *
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

input = sys.argv[1][0:270]

x = data_helpers.filterinput(input)
model = keras.models.load_model('save_GRU_80_2.h5')

layer1 = Flatten()(model.layers[9].output)

model2 = Model(input=[model.input], output=[layer1])
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model2.predict(x))
print(len(model2.predict(x)[0]))

y = model.predict(x)
res = model.predict_proba(x)[0][0]

resultstr = input + "\n"

if round(y) == 1.0:
        resultstr += "Not Spam"
else:
        resultstr += "Spam"

resultstr = resultstr + "\nWith confidence of: " + str(res * 100.0) + "%"

print "---RESULT---" + resultstr + "---RESULT---"
