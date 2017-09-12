from smokeapi import SmokeAPI
import os
from subprocess import list2cmdline

SMOKE = SmokeAPI('fc2611a8b6933e5774e8d1d958ba207cf5b293a90b4bc3b723388b9ba8fe27bc')

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
