"""
Calculates the spearman's rank coefficient and OLS R^2 value for course grades. 
Compares the course grades to overall upper division GPA. 
Plots the courses on a radial plot. 
Courses closest to (1,1) or the origin of the radial plot are most highly correlated. 
"""

import pandas as pd
from matplotlib import rc
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.spatial.distance as sp
from scipy import stats
from pylab import *
import sys

import collections
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import operator

"""
Read in spectral data file which contains students who have taken five mathematics courses 
and at least one upper division course. 
"""
df = pd.read_csv('big_file_excel.csv')
id_df = pd.read_csv('spectral_data.csv')


final_ID_list = id_df['ID'].values.tolist()	#put values of ID from spectral data into list. 
df = df[df['ID'].isin(final_ID_list)]		#Change dataframe to have only students from final_ID_list



#Condition by major
lst_of_math_maj = ['540']
gender = ['F']

#removes '0' before course code and white space after course and subject
df['course'] = df['course'].map(lambda x: x.lstrip('0').lstrip(' ').rstrip(' '))
df['subject'] = df['subject'].map(lambda x: x.rstrip(' '))

#Condition by class standing
df = df[df['AdmitClass'].isin(['UFR', 'USO'])]
#df = df[df['AdmitClass'].isin(['UJR', 'USR'])]


df = df[df['subject'] == "MATH"]


#Only keep math majors
df = df[df['MajorCode'].isin(lst_of_math_maj)]
orig = df
total = len(orig)

#Gender conditioning
df = df[df['Gender'].isin(gender)]




def term_to_float(x):
	"""Function that converts Term (ex: 14W) to float (ex: 14.01) """
	t=x
	a=x
	
	#get term letter
	t = t[-1:]
	
	if(t == 'F'):
		t = 0.4
	elif(t == 'W'):
		t = 0.1
	elif(t == 'S'):
		t = 0.2
	else:
		t = 0.3
	t = t 
	a = a[:-1]
 
	#cast string to float
	a = float(a)
	
	#add year to term
	return (a+t)


counter = collections.Counter(df['course'].values) #Counts all unique courses in dataframe.
common_courses=[]
for key in dict(counter):
	if dict(counter)[key] > 0.01*total:
		common_courses.append(key)
	
#number_of_courses = 30
#common_courses =  dict(sorted(counter.iteritems(), key = operator.itemgetter(1), reverse = True)[:number_of_courses]).keys() #Creates a list of 30 most popular courses. 

"""
Iterates through all courses in common_courses list. 
"""

x_plot = []
y_plot = []
label_course = []
dic_course_coef = {}
dic_GPA_coef = {}
counter_for_courses = 0

for index, course in enumerate(common_courses):

	#Declare lists for Spearman and Persons analysis
	course_grade = []
	GPA_list = []

	#Get all students who have taken specific course
	temp_course = df[df['course'] == (course)] 
	temp_df = df[df['ID'].isin(temp_course['ID'].values.tolist())] #Creates dataframe with specific course

	#Group dataset by ID (unique students)
	grouped = temp_df.groupby('ID')

	"""
	Iterate though each group (unique student) to create list of course grades. 
	"""
	for name, group in grouped:
		#lst_courses = []
		lst_courses = group['course'].values.tolist() #Creates list of all courses taken by unique student. 
		#term_class = 0
		total = 0
		GPA = 0

		"""
		Handles the case when a course only appears once in student's list of all courses. 
		"""
		if lst_courses.count(course) == 1:										#If the course only appears once in the list
			for index, course2 in enumerate(lst_courses):							#Loop through the course list
				if course2 == course:												#If the course is found in the list, add grade to list
					#term_class = group['Term'].values.tolist()[index]
					course_grade.append(group['grade'].values.tolist()[index])

		"""
		Handles the case when a course appears more than once in a student's list of all courses. 
		"""
		if lst_courses.count(course) > 1: 												#If the course appears more than once in the course list
			temp_list = []																#Creates temporary list to hold grade values
			for index1, course3 in enumerate(lst_courses):
				if course3 == course: 													#Finds all instances of course and adds grades to list
					term_class = term_to_float(group['Term'].values.tolist()[index1])
					temp_list.append(group['grade'].values.tolist()[index1])
			if len(temp_list) != 0:
				course_grade.append(sum(temp_list)/float(len(temp_list)))				#Finds average of temp_list and adds final GPA to course_grade list. 
		"""
		Handles the case when a course does not appear in student's list of all courses. 
		"""
		


		"""
		Calculates upper division GPA of unique student. 
		"""

		grade = [] 

		for index2, course4 in enumerate(group['course'].values.tolist()):
			course4 = course4.rstrip('ABCDEFGHIJKLMNOPQRST')					#Removes all letters from the course value
			#change to 99 for upper division
			if course4 > 0:													#If the course is an upper divison course, we add it to grade list 
				grade.append(group['grade'].values.tolist()[index2])

		if len(grade) != 0:
			GPA = sum(grade)/float(len(grade))
		else:
			GPA = 0

		GPA_list.append(GPA)	#adds GPA value to GPA list of all students

	spearman = stats.spearmanr(course_grade, GPA_list)[0]	#Records spearman value based on GPA and course_grade

	"""
	Creates a matrix that contains 1's in the first column, course grades in the second column, and GPA values in the third column. 
	"""
	A = np.zeros(shape= (len(course_grade), 3))
	A[:,0] = 1
	A[:,1] = course_grade
	A[:,2] = GPA_list

	temp_matrix = A
	size = int(len(course_grade)*0.75)	#size is a random fraction of course_grade length. 

	r_sq1 = []	#Empty list to record r squared values
	r_sq = [] 	#For lasso regression

	"""
	Runs a linear regression on a random sample of the data. Collects r squared values. 
	"""
	times = 0
	times2 = 0
	while times < 100:
		np.random.shuffle(temp_matrix)

		set1 = temp_matrix[0:size][:,0:2]
		set2 = temp_matrix[0:size][:,2]
		test_set1 = temp_matrix[size:][:,0:2]
		test_set2 = temp_matrix[size:][:,2]

		clf = Lasso(alpha = 5)
		clf.fit(set1, set2)
		r_sq.append(clf.score(test_set1, test_set2))



		clf1 = LinearRegression(fit_intercept = True)
		clf1.fit(set1, set2)
		r_sq1.append(clf1.score(test_set1, test_set2))
		times = times + 1


	"""
	Outputs the results of the linear regression using spearman's rank and OLS regression. 
	"""
	#print "Course :" , course, "Spearman: ", stats.spearmanr(course_grade, GPA_list)[0], ",    R^2: ", np.asarray(r_sq1).mean()

	label_course.append(course)									#adds course to label_course list for future reference
	x_plot.append(stats.spearmanr(course_grade, GPA_list)[0])	#adds spearman value to x_plot list for future reference
	y_plot.append(np.asarray(r_sq1).mean())						#adds r squared value to y_plot list for future reference

	"""
	Outputs the results of Lasso regression

	"""	
	#if np.asarray(r_sq).mean() > 0:			
		#print "Course: ", course, 'Lasso R-squared',np.asarray(r_sq).mean()
	"""
	Adds the first one hundred courses to a dictionary for future reference. 
	Allows user to keep track of courses and associated grade value
	"""
	if counter_for_courses < 100: 
		dic_course_coef[course] = course_grade
	counter_for_courses += 1

	a = np.asarray([stats.spearmanr(course_grade, GPA_list)[0], np.asarray(r_sq1).mean()])
	dic_GPA_coef[course] = a 	#creates dictionary with course and associated spearman and r squared values

	"""
	Calculates the distance between the x value (spearman's) and the y value (r squared) of an associated course. 
	Used later for generating radial plots
	"""
	distance = []
	for index, course in enumerate(dic_GPA_coef.keys()):
		b = np.asarray([1,1])
		distance.append(sqrt((1-dic_GPA_coef[course][0])**2 + (1-dic_GPA_coef[course][1])**2))

	final_courses = []
	distance_final = []

	"""
	Saves in a final_courses list the courses which are most highly correlated (has minimum distance)
	"""
	for index, value in enumerate(range(0,len(dic_course_coef.keys()))):
		#print distance.index(min(distance))
		final_courses.append(dic_course_coef.keys()[distance.index(min(distance))])
		distance_final.append(min(distance))
		distance[distance.index(min(distance))] = 100

grade_data = pd.DataFrame()
		
all_courses = list(set([course for course in df['course']]))
clf = Lasso(alpha = 0.05)

for course in all_courses:
	
	
clf.fit(set1, set2)
r_sq.append(clf.score(test_set1, test_set2))
		
print clf.coef_
print clf.intercept_

raise SystemExit

"""
Radial Plot code
contains area, r, theta, and colors
"""

final_df = orig[orig['ID'].isin(final_ID_list)]

area = []
new_label_course = []
new_xy = []

xy = zip(x_plot, y_plot)

"""
Does not  considers courses which are  honors,  199, or no longer exist. 
"""
for index, course in enumerate(label_course):
	if course != '199' and course[len(course)-1] != "H" and course != "135B":
		new_label_course.append(course)			#If the course meets the conditions, add it to the new list 
		new_xy.append(xy[index])				#Adds the associated (x,y) value



r = []
theta = []
colors = []

for index, point in enumerate(new_xy):		
	dist = math.hypot(1 - point[0], 1 - point[1])
	r.append(dist)
	#theta.append(2*pi*dist*index*2*(index+sqrt(index)*index**2))
	theta.append(0.3+index+pi)
	colors.append('steelblue')

course_dist_dict = dict(zip(new_label_course, r))

new_r = []
new_course_list = []
new_theta = []

"""
Sorts the r values in increasing order to generate theta values
"""

count = 0
for key, value in sorted(course_dist_dict.iteritems(), key = lambda (k,v):(v,k)):
	new_r.append(value)
	new_course_list.append(key)
	#flower spiral
	#new_theta.append(0.1*pi*count)
	count += 1
	
	
"""
Counts the number of people who have taken the course
"""
for index, course in enumerate(new_course_list): 
	count = 0
	for index2, course2 in enumerate(final_df['course'].values.tolist()):
		if course == course2:
			count += 1
	area.append(count)
	
"""
Calculates theta values so nodes are plotted in spiral fashion
"""
for index, value in enumerate(range(0,len((new_r)))):
	new_theta.append(20+index * (370/len(new_r)))

for index, value in enumerate(new_theta):
	new_value = pi/180 * value
	new_theta[index] = new_value

r = new_r 				#Sets the old r list to the new_r sorted list
theta = new_theta 		#Sets the old theta list to the new_theta list

new_course_list.append('(1,1)')


"""
Sets the color list to orange if the course is a required course
"""

pm_req = ['31A', '31B', '32A', '32B',  '33A', '33B','115A', '131A','131B','132','110A','110B']

final_label_course = []
for index, course in enumerate(new_course_list):
	for index2, course2 in enumerate(pm_req):
		if course == course2:
			colors[index] = 'yellow'
	final_label_course.append('')

for index, course in enumerate(pm_req):
	for index2, course2 in enumerate(new_course_list):
		if course == course2:
			final_label_course[index2] = course


ax = subplot(111, polar=True)
ax.set_ylim(0,1)
ax.set_rlabel_position(0)
plt.thetagrids(range(0))

theta.append(0)
area.append(100)
colors.append('black')
r.append(0)

count = 0 
for axis in zip(theta,r): 
	ax.annotate(new_course_list[count], xy = axis, xytext = (-2, 2),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.3', fc = 'white', alpha = 0.01))
	count += 1


for i in range(len(area)):
	area[i]=area[i]/2
	
c = scatter(theta, r, c=colors, s=area, cmap=cm.hsv)
c.set_alpha(1)

show()
plt.show()

"""
Most popular electives for pm_fr group
"""
pm_elec = []
pm_courses = orig['course'].values.tolist()
for index, course in enumerate(pm_courses):
	if course not in pm_req:
		pm_elec.append(course)

counter = collections.Counter(pm_elec)



