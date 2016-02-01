import pandas as pd
from matplotlib import rc
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.spatial.distance as sp
from scipy import stats
from pylab import *

from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn import decomposition
import sys
import operator
from operator import itemgetter

from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
import collections
from math import sqrt

pm_req = ['115A', '131A', '131B', '132', '110A', '11B']


#(Insert your own data for x_plot and y_plot)
#x_plot is a list of all spearman coeff values
#y_lot is a list of all R^2 values from OLS

""" Example of my label_course, x_plot, y_plot
label_course.append(course)
x_plot.append(stats.spearmanr(course_grade, GPA_list)[0])
y_plot.append(np.asarray(r_sq1).mean())
"""

#label_course is a list of all courses in our set 


xy = zip(x_plot, y_plot)

new_label_course = []
new_xy = []
for index, course in enumerate(label_course):
	if course != '199' and course[len(course)-1] != "H":
		print course
		new_label_course.append(course)
		new_xy.append(xy[index])


r = []
theta = []
colors = []
for index, point in enumerate(new_xy):		
	dist = math.hypot(1 - point[0], 1 - point[1])
	r.append(dist)
	#theta is essentially calculated randomly. This can be changed. 
	theta.append(2*pi*dist*index*2*(index+sqrt(index)*index**2))
	colors.append('blue')

final_label_course = []


#This code erases all labels of courses that are not required (to make visual more clean)
for index, course in enumerate(new_label_course):
	for index2, course2 in enumerate(pm_req):
		if course == course2:
			colors[index] = 'orange'
	final_label_course.append('')

for index, course in enumerate(pm_req):
	for index2, course2 in enumerate(new_label_course):
		if course == course2:
			final_label_course[index2] = course

#This part of the code labels each data point
ax = subplot(111, polar=True)
count = 0 
for axis in zip(theta,r): 
	ax.annotate(new_label_course[count], xy = axis, xytext = (-2, 2),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.3', fc = 'white', alpha = 0.01))
	count += 1

c = scatter(theta, r, c=colors, s=area, cmap=cm.hsv)
c.set_alpha(0.75)

show()
plt.show()