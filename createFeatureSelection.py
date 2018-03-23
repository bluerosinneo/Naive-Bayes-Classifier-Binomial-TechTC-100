
# Python version: 3.5.4
# Numpy version: 1.14.0
# SciKit-Learn version : 0.19.1

# Data Information
# http://techtc.cs.technion.ac.il/techtc100/techtc100.html
# Evgeniy Gabrilovich and Shaul Markovitch
# "Text Categorization with Many Redundant Features: Using Aggressive Feature Selection to Make SVMs Competitive with C4.5"
# The 21st International Conference on Machine Learning (ICML), pp. 321-328, Banff, Alberta, Canada, July 2004
# id1:6920 id2:8366 from techtc100_preprocessed.zip

import numpy as np
# from sklearn.model_selection import train_test_split

# consideration is only given to the "index" of features
# no consideration is given to what the actual feature "word" is
# note that files index feature starting with 1
# we will index features starting with 0
with open('features.idx', 'r') as f:
	Features = [line.split()[0]
		for line in f.readlines()
		if not (line.startswith("#"))  # ignore lines that start with # (comments)
		and len(line.split())>0] # ignore lines that just white space
f.closed

X = [] # feature matrix contains the number of instances of each feature (integer)
XoneZero = [] # feature matrix contains a boolean of if the particular feature is preasent (0/1 "boolean")
y = [] # class label vector
with open('vectors.dat', 'r') as f:
	for line in f:
		if not (line.startswith('#')): # ignore lines that start with # (comments)
			line = line.split()
			if (int(line[0])==1): # first element in each line is the class label, append to y
				y.append(1)
			else:
				y.append(0)
			x = [0]*len(Features) # temporary feature vector for individual sample (int)
			xOne = [0]*len(Features) # temporary feature vecotr for indeividual sample ("boolean")
			for pair in line[1:]: # remember that first element in line is class feature
				pairSplit = pair.split(':') # pairs look like "feature index":"number in sample"
				if (pairSplit[0] in Features): # if feature index is present append to x and xOne
					x[Features.index(pairSplit[0])]=int(pairSplit[1]) # use indexing starting with 0 not 1
					xOne[Features.index(pairSplit[0])]=1
			X.append(x) # append feature vector to feature matrix
			XoneZero.append(xOne) # append feature vector to feature matrix
f.closed

# convert to numpy arrays
X = np.array(X)
XoneZero = np.array(XoneZero)
y = np.array(y)

print ()
print ("number of features 'raw data'")
print ("samples by number of features: {}".format(X.shape))
print ()

# work on selecting features that apear at least in 2 samples
atLeastTwoTimes = XoneZero.sum(axis=0)>=2 # create boolean (True/False) vector corresponding to features
X_leastTwo = X[:,atLeastTwoTimes]
XoneZero_leastTwo = XoneZero[:,atLeastTwoTimes]
print ()
print ("number of features after removing features that don't apear at least 2 times")
print ("samples by number of features: {}".format(X_leastTwo.shape))
print ()

# work on selecting features that apear at least in 5 samples
atLeastFiveTimes = XoneZero.sum(axis=0)>=5 # create boolean (True/False) vector corresponding to features
X_leastFive = X[:,atLeastFiveTimes]
XoneZero_leastFive = XoneZero[:,atLeastFiveTimes]
print ()
print ("number of features after removing features that don't apear at least 5 times")
print ("samples by number of features: {}".format(X_leastFive.shape))
print ()

# work on selecting features that apear at least in 10 samples
atLeastTen = XoneZero.sum(axis=0)>=10 # create boolean (True/False) vector corresponding to features
X_leastTen = X[:,atLeastTen]
XoneZero_leastTen = XoneZero[:,atLeastTen]
print ()
print ("number of features after removing features that don't apear at least 10 times")
print ("samples by number of features: {}".format(X_leastTen.shape))
print ()

# want to reshape y (class labels) so that it is easier to wrap up for later use
n,dNotUsed = X.shape
y = y.reshape((n,1))

data = np.concatenate((y,X),axis=1)
data = np.concatenate((data,XoneZero),axis=1)
np.savetxt('allFeatures.txt', data, delimiter=',', fmt='%1.0f')

data = np.concatenate((y,X_leastTwo),axis=1)
data = np.concatenate((data,XoneZero_leastTwo),axis=1)
np.savetxt('atLeastTwo.txt', data, delimiter=',', fmt='%1.0f')

data = np.concatenate((y,X_leastFive),axis=1)
data = np.concatenate((data,XoneZero_leastFive),axis=1)
np.savetxt('atLeastFive.txt', data, delimiter=',', fmt='%1.0f')

data = np.concatenate((y,X_leastTen),axis=1)
data = np.concatenate((data,XoneZero_leastTen),axis=1)
np.savetxt('atLeastTen.txt', data, delimiter=',', fmt='%1.0f')