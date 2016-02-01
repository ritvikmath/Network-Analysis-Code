import numpy as np
import pandas as pd

mefr = pd.read_csv('mefr.csv')
metr = pd.read_csv('metr.csv')
amfr = pd.read_csv('amfr.csv')
amtr = pd.read_csv('amtr.csv')
pmfr = pd.read_csv('pmfr.csv')
pmtr = pd.read_csv('pmtr.csv')

l=[mefr, metr, amfr, amtr, pmfr, pmtr]

maxlist = []
avglist=[]

for group in l:
	dists = group['dist']
	diffs = []
	for i in range(1,len(dists)):
		diffs.append(dists[i] - dists[i-1])
	maxlist.append(max(diffs))
	avglist.append(np.mean(diffs))
		
