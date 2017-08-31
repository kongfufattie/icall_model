import pandas as pd
import util
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import classification_report

#build 7's model after case study
data7=pd.read_excel('201707_label new.xlsx')
#data7=data7[data7['是否接通']==1]
#data7=data7[['存折计划' in c for c in data7['租机计划']]]
#data7['跪舔']=data7['跪舔']-data7['label']#这个feature使f1-score增加0.02
data7.loc[data7['跪舔']>=1,'跪舔']=1
data7=util.preprocessing(data7)
X=features=util.extractFeatures(data7)
y=data7['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model=GradientBoostingRegressor()
model.fit(X_train, y_train)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
y_pre=model.predict(X_test)
a=pd.DataFrame({"pre":y_pre,"y_test":y_test}).merge(data7,how='left',left_index=True,right_index=True)[['pre','跪舔','y_test']]
for k in range(50,105,5):
    k=k/100
    a['pre_weight_{}'.format(k)]=k*a.pre+(1-k)*a['跪舔']
    util.showReport(a['pre_weight_{}'.format(k)],a.y_test,30)
#util.showFigure1(y_pre,y_test)
#util.showPRfigure(y_pre,y_test)
#util.showReport(y_pre,y_test,20)
'''
#predict 8 with 7's model
data7=pd.read_excel('201707_label.xlsx')
data7=data7[data7['是否接通']=='是']
data7['跪舔']=data7['跪舔']-data7['label']
data7.loc[data7['跪舔']>=1,'跪舔']=1
data8=pd.read_excel('201708.xlsx')
data8.loc[data8['跪舔']>=1,'跪舔']=1
data=pd.concat((data7,data8), axis=0, join='inner',ignore_index=True)
data=util.preprocessing(data)
X=features=util.extractFeatures(data)
X_train=X[:data7.shape[0]]
X_test=X[data7.shape[0]:]
y_train=data7['label']
model=GradientBoostingRegressor()
model.fit(X_train, y_train)
print(model.score(X_train,y_train))
y_pre=model.predict(X_test)
data8['label']=y_pre
data8['label']=0.9*data8['label']+0.1*data8['跪舔']
data8.sort_values(by='label',ascending=False,inplace=True)
data8.to_csv('201708_score_1.csv')

#slice
import math
ds=[]
ds.append(data8[:6000].sort_values(by='客户名称',ascending=False))
ds.append(data8[6000:20000].sort_values(by='客户名称',ascending=False))
count= math.floor((data8.shape[0]-20001)/4)
for i in range(4):
    if i==3:
        ds.append(data8[20000+i*count:].sort_values(by='客户名称',ascending=False))
    else:
        ds.append(data8[20000+i*count:20000+(i+1)*count].sort_values(by='客户名称',ascending=False))
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