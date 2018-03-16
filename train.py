import csv
import numpy as np
from sklearn import svm

	
filename = 'D:\Documents\Python\sum1\DATA.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x).astype('float')
data_rows=data.shape[0];
data_cols=data.shape[1];

#print(data[1:3,:])

#
train_data=data[0:int(data_rows*0.7),0:3];
test_data=data[int(data_rows*0.7):(data_rows-1),0:3];

#output
train_data_o=data[0:int(data_rows*0.7),4];
test_data_o=data[int(data_rows*0.7):(data_rows-1),4];

clf = svm.SVC()
clf.fit(train_data,train_data_o)
print(clf.predict(test_data[0:3,]))
print(test_data_o[0:3,]) 