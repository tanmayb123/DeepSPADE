from smokeapi import SmokeAPI
import os
from subprocess import list2cmdline

SMOKE = SmokeAPI('')

SMOKE.per_page = 1

def runStep(lastBody):
	latestresult = SMOKE.fetch('posts/search')['items'][0]['body'].replace("\n", " ")
	if latestresult != lastBody:
		prediction = os.popen(list2cmdline(["python", "getresult.py", latestresult])).read()
		os.popen(list2cmdline(["python", "a.py", prediction]))
	Timer(3.0, runStep, latestresult)

runStep("")

while (True):
	i = 9
