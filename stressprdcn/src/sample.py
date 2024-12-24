
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

file1=open("data1.txt",mode="r",encoding="utf-8")
print("read words")
text=file1.read()
print(len(text))
rdwrds=text.split("\n")
dataset=[]
for i in rdwrds:
    ii=i.split(',')
    if len(ii)>3:
        dataset.append(ii)
print(dataset)
print(len(dataset[0]))

features=[]
labels=[]
ii=0
for i in dataset:

    row=[]
    for ii in range(0,31):
        row.append(int(i[ii]))
    features.append(row)
    labels.append(int(i[31]))
X=features
y=labels

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)



#Defining the machine learning models

model2 = DecisionTreeClassifier(max_depth = 2)
model3 = SVC()


model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)

cm_DecisionTree = confusion_matrix(y_test, y_pred2)
cm_SupportVectorClass = confusion_matrix(y_test, y_pred3)


#10-fold cross-validation
kfold = model_selection.KFold(n_splits=10)

result2 = model_selection.cross_val_score(model2, X_train, y_train, cv=kfold)
result3 = model_selection.cross_val_score(model3, X_train, y_train, cv=kfold)


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Printing the accuracies achieved in cross-validation
print('Accuracy and CM of Decision Tree Model = ')
cm=confusion_matrix(y_test, y_pred2)
ac=accuracy_score(y_test, y_pred2)
print(cm)
print(ac)


print('Accuracy and CM of Support Vector Machine ')
cm=confusion_matrix(y_test, y_pred3)
ac=accuracy_score(y_test, y_pred3)
print(cm)
print(ac)

#Defining Hybrid Ensemble Learning Model
# create the sub-models
estimators = []



#Defining 5 Decision Tree Classifiers
model16 = DecisionTreeClassifier(max_depth = 3)
estimators.append(('cart1', model16))
model17 = DecisionTreeClassifier(max_depth = 4)
estimators.append(('cart2', model17))
model18 = DecisionTreeClassifier(max_depth = 5)
estimators.append(('cart3', model18))
model19 = DecisionTreeClassifier(max_depth = 2)
estimators.append(('cart4', model19))
model20 = DecisionTreeClassifier(max_depth = 3)
estimators.append(('cart5', model20))

#Defining 5 Support Vector Classifiers
model21 = SVC(kernel = 'linear')
estimators.append(('svm1', model21))
model22 = SVC(kernel = 'poly')
estimators.append(('svm2', model22))
model23 = SVC(kernel = 'rbf')
estimators.append(('svm3', model23))
model24 = SVC(kernel = 'rbf')
estimators.append(('svm4', model24))
model25 = SVC(kernel = 'linear')
estimators.append(('svm5', model25))


# Defining the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

#Confisuin matrix
cm_HybridEnsembler = confusion_matrix(y_test, y_pred)

# confusion matrix of hybrid ensemble learning model


kfold = model_selection.KFold(n_splits=10)
results = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)


print('Accuracy and CM of Hybrid SVM and DT')

cm=confusion_matrix(y_test, y_pred)
ac=accuracy_score(y_test, y_pred)
print(cm)
print(ac)
