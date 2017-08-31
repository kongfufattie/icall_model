import pandas as pd
import util
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.externals import joblib

#build 6's model after case study
data6=pd.read_excel('201706.xlsx')
#data6=data6[data6['是否接通']==1]
#data6=data6[['存折计划' in c for c in data6['租机计划']]]
data6=util.preprocessing(data6)
X=features=util.extractFeatures(data6)
y=data6['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model=RandomForestRegressor(n_jobs=-1)
model.fit(X_train, y_train)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
y_pre=model.predict(X_test)
util.showFigure1(y_pre,y_test)
util.showPRfigure(y_pre,y_test)
util.showReport(y_pre,y_test,20)
print(model.feature_importances_)
'''
#predict 7 with 6's model
data6=pd.read_excel('201706.xlsx')
data6=data6[data6['是否接通']==1]
data7=pd.read_excel('201707.xlsx')
data=pd.concat((data6,data7), axis=0, join='inner',ignore_index=True)
data=util.preprocessing(data)
X=features=util.extractFeatures(data)
y=data['label']
X_train=X[:data6.shape[0]]
X_test=X[data6.shape[0]:]
y_train=y[:data6.shape[0]]
model=GradientBoostingRegressor()
model.fit(X_train, y_train)
print(model.score(X_train,y_train))
y_pre=model.predict(X_test)
data7['label']=y_pre
data7.sort_values(by='label',ascending=False,inplace=True)
data7.to_csv('201707_score_1.csv')

#slice
import math
ds=[]
ds.append(data7[:5000].sort_values(by='客户名称',ascending=False))
ds.append(data7[5000:20000].sort_values(by='客户名称',ascending=False))
count= math.floor((data7.shape[0]-20001)/4)
for i in range(4):
    if i==3:
        ds.append(data7[20000+i*count:].sort_values(by='客户名称',ascending=False))
    else:
        ds.append(data7[20000+i*count:20000+(i+1)*count].sort_values(by='客户名称',ascending=False))
ds1=[]
ds1.append(ds[0][:1500])
ds1.append(ds[0][1500:3000])
ds1.append(ds[0][3000:4500])
ds1.append(ds[0][4500:])
for i in range(4):
    ds1[i].to_csv('s1_{}.csv'.format(i+1))
for i in range(1,6):
    ds[i].to_csv('s{}.csv'.format(i+1))
#check mean score
for d in ds:
    print(d.label.mean())
for d in ds1:
    print(d.label.mean())
'''