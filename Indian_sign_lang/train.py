import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict=pickle.load(open('data.pickle1','rb'))

a=np.array(data_dict['data'],dtype=object)
l=np.array([len(a[i]) for i in range(len(a))])
width=l.max()
data=[]
for i in range(len(a)):
    if len(a[i])!=width:
        x=np.pad(a[i],(0,width-len(a[i])),'constant',constant_values=0)
    else:
        x=a[i]
    data.append(x)

data=np.asarray(data)
labels=np.asarray(data_dict['labels'],dtype=object)


x_train, x_test, y_train, y_test=train_test_split(data,labels,test_size=0.3,shuffle=True,stratify=labels)

model=RandomForestClassifier()

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

score= accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f=open('model.p','wb')
pickle.dump({'model':model},f)
f.close()