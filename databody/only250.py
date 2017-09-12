import os

filename = "pos.txt"

opened = open(filename, "r").read().split("\n")

rows = []

for i in opened:
	rows.append(i[0:270])

os.popen("rm pos.txt")
open("pos.txt", "w").write('\n'.join(rows))
