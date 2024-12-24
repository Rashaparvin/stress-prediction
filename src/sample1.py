#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn. preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0, stratify = y)
#
# # Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# #Defining the machine learning models
# model1 = LogisticRegression()
# model2 = DecisionTreeClassifier(max_depth = 2)
# model3 = SVC()
# model4 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# model5 = GaussianNB()
#
# #Training the machine learning models
# model1.fit(X_train, y_train)
# model2.fit(X_train, y_train)
# model3.fit(X_train, y_train)
# model4.fit(X_train, y_train)
# model5.fit(X_train, y_train)
#
# #Making the prediction
# y_pred1 = model1.predict(X_test)
# y_pred2 = model2.predict(X_test)
# y_pred3 = model3.predict(X_test)
# y_pred4 = model4.predict(X_test)
# y_pred5 = model5.predict(X_test)
#
# #Confusion matrix
# cm_LogisticRegression = confusion_matrix(y_test, y_pred1)
# cm_DecisionTree = confusion_matrix(y_test, y_pred2)
# cm_SupportVectorClass = confusion_matrix(y_test, y_pred3)
# cm_KNN = confusion_matrix(y_test, y_pred4)
# cm_NaiveBayes = confusion_matrix(y_test, y_pred5)
#
#
# #10-fold cross-validation
# kfold = model_selection.KFold(n_splits=10, random_state = 0)
# result1 = model_selection.cross_val_score(model1, X_train, y_train, cv=kfold)
# result2 = model_selection.cross_val_score(model2, X_train, y_train, cv=kfold)
# result3 = model_selection.cross_val_score(model3, X_train, y_train, cv=kfold)
# result4 = model_selection.cross_val_score(model4, X_train, y_train, cv=kfold)
# result5 = model_selection.cross_val_score(model5, X_train, y_train, cv=kfold)
#
# #Printing the accuracies achieved in cross-validation
# print('Accuracy of Logistic Regression Model = ',result1.mean())
# print('Accuracy of Decision Tree Model = ',result2.mean())
# print('Accuracy of Support Vector Machine = ',result3.mean())
# print('Accuracy of k-NN Model = ',result4.mean())
# print('Accuracy of Naive Bayes Model = ',result5.mean())
#


#Defining Hybrid Ensemble Learning Model
# create the sub-models
estimators = []

#Defining 5 Logistic Regression Models
model11 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic1', model11))
# model12 = LogisticRegression(penalty = 'l2', random_state = 0)
# estimators.append(('logistic2', model12))
# model13 = LogisticRegression(penalty = 'l2', random_state = 0)
# estimators.append(('logistic3', model13))
# model14 = LogisticRegression(penalty = 'l2', random_state = 0)
# estimators.append(('logistic4', model14))
# model15 = LogisticRegression(penalty = 'l2', random_state = 0)
# estimators.append(('logistic5', model15))

#Defining 5 Decision Tree Classifiers
model16 = DecisionTreeClassifier(max_depth = 3)
estimators.append(('cart1', model16))
# model17 = DecisionTreeClassifier(max_depth = 4)
# estimators.append(('cart2', model17))
# model18 = DecisionTreeClassifier(max_depth = 5)
# estimators.append(('cart3', model18))
# model19 = DecisionTreeClassifier(max_depth = 2)
# estimators.append(('cart4', model19))
# model20 = DecisionTreeClassifier(max_depth = 3)
# estimators.append(('cart5', model20))

#Defining 5 Support Vector Classifiers
model21 = SVC(kernel = 'linear')
estimators.append(('svm1', model21))
# model22 = SVC(kernel = 'poly')
# estimators.append(('svm2', model22))
# model23 = SVC(kernel = 'rbf')
# estimators.append(('svm3', model23))
# model24 = SVC(kernel = 'rbf')
# estimators.append(('svm4', model24))
# model25 = SVC(kernel = 'linear')
# estimators.append(('svm5', model25))

#Defining 5 K-NN classifiers
model26 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
estimators.append(('knn1', model26))
# model27 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# estimators.append(('knn2', model27))
# model28 = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
# estimators.append(('knn3', model28))
# model29 = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 1)
# estimators.append(('knn4', model29))
# model30 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 1)
# estimators.append(('knn5', model30))

#Defining 5 Naive Bayes classifiers
model31 = GaussianNB()
estimators.append(('nbs1', model31))
# model32 = GaussianNB()
# estimators.append(('nbs2', model32))
# model33 = GaussianNB()
# estimators.append(('nbs3', model33))
# model34 = GaussianNB()
# estimators.append(('nbs4', model34))
# model35 = GaussianNB()
# estimators.append(('nbs5', model35))

# Defining the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

#Confisuin matrix
cm_HybridEnsembler = confusion_matrix(y_test, y_pred)

# confusion matrix of hybrid ensemble learning model

#Cross-Validation
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
results = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)
print(results.mean())