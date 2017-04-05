# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:28:02 2017

@author: don
"""
import os
import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import collections as counter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
os.chdir('C:\\Users\don\Desktop\kNN classifiers\Giant Table\Pipeline')


names=['Direction/Sensor/Bill', 'Detection', 'tach1', 'tach2', 'tach3', 'tach4', 
'tach5', 'tach6', 'tach7', 'tach8', 'tach9', 'tach10', 'tach11', 'tach12', 
'tach13', 'tach14', 'tach15', 'tach16', 'tach17', 'tach18', 'tach19', 
'tach20', 'tach21', 'tach22', 'tach23', 'tach24', 'tach25', 'tach26', 
'tach27', 'tach28', 'tach29', 'tach30', 'tach31', 'tach32', 'tach33', 
'tach34', 'tach35', 'tach36', 'tach37', 'tach38', 'tach39', 'tach40', 
'tach41', 'tach42', 'tach43', 'tach44', 'tach45', 'tach46', 'tach47', 
'tach48', 'tach49', 'tach50', 'tach51', 'tach52', 'tach53', 'tach54', 
'tach55', 'tach56', 'tach57', 'tach58', 'tach59', 'tach60', 'tach61', 
'tach62', 'tach63', 'tach64', 'tach65', 'tach66', 'tach67', 'tach68', 
'tach69', 'tach70', 'tach71', 'tach72', 'tach73', 'tach74', 'tach75', 
'tach76', 'tach77', 'tach78', 'tach79', 'tach80', 'tach81', 'tach82', 
'tach83', 'tach84', 'tach85', 'tach86', 'tach87', 'tach88', 'tach89', 
'tach90']


df = pd.read_csv('mex_ld_lu2.csv', header=0, names=names)


# create design matrix X and target vector y
X = np.array(df.ix[:,1:93]) 	# end index is exclusive (Tachs 0-90)
y = np.array(df['Direction/Sensor/Bill'])

#Xnew = SelectKBest(chi2, k=23).fit_transform(X, y) #Best-Feature-Selection. 

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

#PIPELINE: RndFrst
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', RandomForestClassifier())
])

# fitting the model
clf.fit(X_train, y_train)

# predict the response
pred = clf.predict(X_test)

# evaluate accuracy
print("Random Forest Accuracy: " + str(accuracy_score(y_test, pred)))

########################################################################

#PIPELINE: Exrtemely Random Forest
clf2 = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', ExtraTreesClassifier(n_estimators=10, max_depth=None,
     min_samples_split=2, random_state=42))
])

# fitting the model
clf2.fit(X_train, y_train)

# predict the response
pred2 = clf2.predict(X_test)

# evaluate accuracy
print("Extra Random Forest Accuracy: " + str(accuracy_score(y_test, pred2)))


########################################################################
clf3 = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
max_depth=1, random_state=20))
])

# fitting the model
clf3.fit(X_train, y_train)

# predict the response
pred3 = clf3.predict(X_test)

# evaluate accuracy
print("Gradient Boosting Accuracy: " + str(accuracy_score(y_test, pred3)))


########################################################################
clf4 = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', LogisticRegression(random_state=1))])

# fitting the model
clf4.fit(X_train, y_train)

# predict the response
pred4 = clf4.predict(X_test)

# evaluate accuracy
print("Logistic Regression Accuracy: " + str(accuracy_score(y_test, pred4)))


########################################################################
clf5 = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
  ('classification', GaussianNB())])

# fitting the model
clf5.fit(X_train, y_train)

# predict the response
pred5 = clf5.predict(X_test)

# evaluate accuracy
print("Logistic Regression Accuracy: " + str(accuracy_score(y_test, pred5)))

eclf = VotingClassifier(estimators=[('lr', clf4), 
                                    ('rf', clf), 
                                    ('gnb', clf5),
                                    ('erf', clf2),
                                    ('gb', clf3)],
                                    voting='hard')


for clf, label in zip([clf, clf2, clf3, clf4, clf5, eclf], ['Random Forest', 
'Extra Random Forest', 'Gradient Boosting',  
'Logistic Regression', 'naive Bayes', 
'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))