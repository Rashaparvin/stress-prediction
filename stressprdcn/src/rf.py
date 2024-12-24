
from sklearn.model_selection import train_test_split

file1=open("data1.txt",mode="r",encoding="utf-8")
print("read words")
text=file1.read()
print(len(text))
rdwrds=text.split("\n")
dataset=[]
for i in rdwrds:
    ii=i.split(',')
    if len(i)>3:
        lis=[]
        for iii in ii:
            lis.append(int(iii))

        dataset.append(lis)
print(dataset)
print(len(dataset[0]))

features=[]
labels=[]
ii=0
for i in dataset:

    row=[]
    for ii in range(0,31):
        row.append(int(i[ii]))
    # if row not in features:
    features.append(row)
    labels.append(int(i[31]))


print("0",labels.count(0))
print("1",labels.count(1))


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 0)

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)

model = RandomForestClassifier()
model.fit(features, labels)

Y_pred = model.predict(test_features)

print(Y_pred)
print(test_labels)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print('Accuracy and CM of RF')
cm=confusion_matrix(Y_pred, test_labels)
ac=accuracy_score(Y_pred, test_labels)
print(cm)
print(ac)

def predictfn(val):
    y=model.predict(val)
    return y
