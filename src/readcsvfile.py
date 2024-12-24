import csv
from src.db_connection import *

file = open('dataset2.csv')

csvreader = csv.reader(file)

header = []
header = next(csvreader)
#print(header)
rows = []
i=1
for row in csvreader:
                rows.append(row)

                #print(row[0])
                row1=row[0].replace('"',"")
                row1=row1.split('\t')
                print(len(row1),row1)
                roww=row1[5:36]
                # print(row)
                #print(roww)
                roww.append(row1[-1])
                val=roww
                #print(len(roww))
                if "" not in val:
                    qry="INSERT INTO stressdata VALUES(NULL,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    val=roww
                    iud(qry,roww)
                else:
                    print("missing data")

qry="SELECT * FROM `stressdata`"

con = pymysql.connect(host="localhost", user="root", password="", port=3306, db="strsprdcn")
cmd = con.cursor()
cmd.execute(qry)
ress = cmd.fetchall()

row_headers = [x[0] for x in cmd.description]
#print(row_headers)
distinct_list=[]
for i in range(1,len(row_headers)):
        qry = "SELECT DISTINCT `"+row_headers[i]+"` FROM `stressdata` ORDER BY `"+row_headers[i]+"`"
        res = selectall2(qry)
        reslis=[]
        for ii in res:
                reslis.append(str(ii[0]).lower())
        distinct_list.append(reslis)
#print(row_headers)

#print(distinct_list)
#print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
for i in distinct_list:
    print(len(i),"##",i)
# for i in distinct_list:
#         #print(i)

result=[]
for i in ress:
        #print(len(i))
        row=[]
        for ii in range(1,33):
                if ii==3:
                        row.append(i[ii])
                elif ii==32:
                        row.append(i[ii])
                else:
                        chr=str(i[ii]).lower()
                        #print(chr)
                        lis=distinct_list[ii-1]
                        #print(lis)
                        pos=lis.index(chr)
                        row.append(pos)
        result.append(row)

import csv

fields = []
for i in range(1,len(row_headers)):
        fields.append(row_headers[i])

# data rows of csv file
rows = result
filename = "dataset1.csv"

with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)
#
