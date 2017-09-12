# -*- coding: utf-8 -*-
import data_helpers
import keras
import sys
from keras.models import Model

reload(sys)
sys.setdefaultencoding('utf-8')

x = data_helpers.filterinput(sys.argv[1])
model = keras.models.load_model('./save_tmp.h5')
newmodel = Model(model.input, model.layers[5].output)

y = model.predict(x)
y2 = newmodel.predict(x)[0][0:len(sys.argv[1].split())]

resultstr = ""

if round(y) == 1.0:
	resultstr = "NOT SPAM"
else:
	resultstr = "SPAM"

index = -1
biggestScore = -1000
biggestWord = ""
for i in y2:
	index += 1
	if biggestScore < i:
		biggestScore = i
		biggestWord = sys.argv[1].split()[index]

resultstr = resultstr + "\nMOST IMPORTANT WORD IN DECISION: \"" + biggestWord + "\""

print "---RESULT---" + resultstr + "---RESULT---"
