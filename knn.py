

# out=prep(lis)



import numpy as np

from collections import Counter

features=[]
labels=[]
ii=0



class CKNN:

    def __init__(self):
        self.accurate_predictions = 0
        self.total_predictions = 0
        self.accuracy = 0.0
        ##########

        lines=[]

        file1 = open(r"C:\Users\Rasha\PycharmProjects\stressprdcn\src\data1.txt", mode="r", encoding="utf-8")
        print("read words")
        text = file1.read()
        print(len(text))
        rdwrds = text.split("\n")
        dataset = []
        for i in rdwrds:
            ii = i.split(',')
            if len(i) > 3:
                lis = []
                for iii in ii:
                    lis.append(int(iii))

                dataset.append(lis)
        print(dataset)
        print(len(dataset[0]))

        features = []
        labels = []
        ii = 0
        for i in dataset:

            row = []
            for ii in range(0, 31):
                if ii != 1 and ii != 3:
                    row.append(int(i[ii]))
            # if row not in features:
            features.append(row)
            labels.append(int(i[31]))

        # for i in range(1,len(training_data1)):
        #     training_data.append(training_data1[i][0:30])
        #     lines.append(training_data1[i][31])
        # print("training_data",training_data)
        training_data=features
        lines=labels
        self.training_set= { '0':[],'1':[]}

        #Split data into training and test for cross validation
        #training_data = lbls[: len(lbls)]
        test_data = []#[-int(test_size * len(dataset)):]

        #Insert data into the training set
        cnt=0

        for record in training_data:
            st=str(lines[cnt])
            cnt+=1


            self.training_set[st[-1]].append( record[:])
            self.training_set[st].append( record[:])

    #########

    def predict(self,  to_predict, k = 1):
        # print(to_predict,training_data['6'][0])
        # if len(training_data) >= k:
        #     print("K cannot be smaller than the total voting groups(ie. number of training data points)")
        #     return

        distributions = []
        for group in self.training_set:
            i=0
            # print(group,'group')
            for features in self.training_set[group]:

                euclidean_distance = np.linalg.norm(np.array(features)- np.array(to_predict))
                if  group=='6':
                    # print('hi',euclidean_distance,training_data[group],len(training_data[group]),len(to_predict),i)
                    i+=1
                distributions.append([euclidean_distance, group])

        # print(distributions)
        results = [i[1] for i in sorted(distributions)[:k]]
        result = Counter(results).most_common(1)[0][0]
        # print("rs",results,self.training_set.keys())
        confidence = Counter(results).most_common(1)[0][1]/k

        return result, confidence



def prep(aa):
    # feat=glcm_feat(filename)
    # aa = [0,0,18,0,3,2,1,2,1,2,2,2,2,3,3,2,1,2,1,1,1,1,2,2,2,2,2,2,2]
    print(len(aa),"===========")
    knn = CKNN()
    res=knn.predict(aa)#training_set['6'][1])

    return res


# print(prep())
