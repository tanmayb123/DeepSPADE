# -*- coding: utf-8 -*-
import data_helpers
import keras
import sys
from keras.utils import plot_model

reload(sys)
sys.setdefaultencoding('utf-8')

input = sys.argv[1][0:270]

x = data_helpers.filterinput(input)
#ensemble3_1
model = keras.models.load_model('save_ensemble3_1.h5')
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

y = model.predict(x)

#resultstr = input + "\n"
resultstr = ""

if round(y) == 1.0:
        resultstr += "Not Spam"
else:
        resultstr += "Spam"

print "---RESULT---" + resultstr + "---RESULT---"
