
# Python version: 3.5.4
# Numpy version: 1.14.0
# SciKit-Learn version : 0.19.1

# createFeatureSelection.py should be run before running this file

import numpy as np
from sklearn.model_selection import train_test_split

def testBayes(X,y):
	Cn = len(np.unique(y)) # capter number of class labels
	sizeOfAcc = 1500 # number of times to capture accuracy
	acc = np.zeros(sizeOfAcc, dtype=float) # vector to store accuracies
	for times in range(0,sizeOfAcc):
		X_train, X_test, y_train, y_test = train_test_split(X, y) # split into training and test set
		n_train = len(X_train)
		n_test = len(X_test)

		# X_train/X_text should all be even
		# splitting into integer feature matrix (_Int) and boolean (0/1) feature matrix (_OneZero)
		n_doNotUse, shouldBeEven = X_train.shape
		shouldBeEven=int(shouldBeEven/2)
		X_train_OneZero = X_train[:,shouldBeEven:]
		X_test_OneZero = X_test[:,shouldBeEven:]
		n,d = X_train_OneZero.shape


		featureGivenClass = np.zeros((Cn,d), dtype=float)
		classProb = np.zeros(Cn, dtype=float)

		for i in range(0,Cn):
			Cin = (X_train_OneZero[y_train==i].sum(axis = 0)+1).sum()
			classProb[i] = len(X_train_OneZero[y_train==i]) # number of feature instances
			featureGivenClass[i]=(X_train_OneZero[y_train==i].sum(axis=0)+1)*(1/Cin)
		classProb = classProb/classProb.sum() # divide by total number of samples in training set

		# Please refer to the wordpress post for vectorization
		Posterior=np.add(np.log(classProb),np.dot(X_test_OneZero,np.log(np.transpose(featureGivenClass)))+np.dot((1-X_test_OneZero),np.log(np.transpose((1-featureGivenClass)))))

		acc[times]=((Posterior.argmax(axis=1)==y_test).sum()/n_test)

	return acc

data =  np.genfromtxt('allFeatures.txt',delimiter=',',dtype='float')
y = data[:,0]
X = data[:,1:]
acc = testBayes(X,y)
print ()
print ("All Features")
print (acc.std())
print (acc.mean())


data =  np.genfromtxt('atLeastTwo.txt',delimiter=',',dtype='float')
y = data[:,0]
X = data[:,1:]
acc = testBayes(X,y)
print ()
print ("At Least in Two:")
print (acc.std())
print (acc.mean())

data =  np.genfromtxt('atLeastFive.txt',delimiter=',',dtype='float')
y = data[:,0]
X = data[:,1:]
acc = testBayes(X,y)
print ()
print ("At Least in Five")
print (acc.std())
print (acc.mean())


data =  np.genfromtxt('atLeastTen.txt',delimiter=',',dtype='float')
y = data[:,0]
X = data[:,1:]
acc = testBayes(X,y)
print ()
print ("At Least in Ten")
print (acc.std())
print (acc.mean())