import sys
import os

res = os.popen("python samplefunc.py \"" + sys.argv[1] + "\"").read().split("---RESULT---")
res.pop()
print res.pop()
