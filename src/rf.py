
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
        if ii!=1 and ii!=3:
            row.append(int(i[ii]))
    # if row not in features:
    features.append(row)
    labels.append(int(i[31]))

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 0)


from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier()
model.fit(features, labels)

Y_pred = model.predict(test_features)

print(Y_pred)
print(test_labels)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print('Accuracy and CM of RF: ')
cm=confusion_matrix(Y_pred, test_labels)
ac=accuracy_score(Y_pred, test_labels)
print(cm)
print(ac)

print(" Random forest is the best algorithm for stress prediction... ")


def predictfn(val):
    y=model.predict([val])
    return y[0]
