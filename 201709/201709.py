import pandas as pd
import util
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import classification_report
'''
#build 7's model with undersampling then predit 8`s data, score is bad(-0.85) but f1-score of 1 is 0.33(better than former 0.28)
data7=pd.read_excel('201707_label new.xlsx')
data7=data7[data7['是否接通']=='是']
data7=data7.sort_values(by='label',ascending=False)[:9000]
data8=pd.read_excel('8月租机到期数据-结果.xlsx')
data=pd.concat((data7,data8), axis=0, join='inner',ignore_index=True)
data=util.preprocessing(data)
X=util.extractFeatures(data)
y=data['label']
model=GradientBoostingRegressor()
X_train=X[:9000]
y_train=y[:9000]
X_test=X[9000:]
y_test=y[9000:]
model.fit(X_train, y_train)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
y_pre=model.predict(X_test)
util.showFigure1(y_pre,y_test)
util.showPRfigure(y_pre,y_test)
util.showReport(y_pre,y_test,50)
'''

#predict 9 with 8's model
data8=pd.read_excel('8月租机到期数据-结果.xlsx')
data8=data8[data8['是否接通']==1]
data8['跪舔']=data8['跪舔']-data8['label']
data8.loc[data8['跪舔']>=1,'跪舔']=1
data9=pd.read_csv('201709-租机到期数据.csv',encoding='gbk')
data9.loc[data9['跪舔']>=1,'跪舔']=1
data=pd.concat((data8,data9), axis=0, join='inner',ignore_index=True)
data=util.preprocessing(data)
X=features=util.extractFeatures(data)
X_train=X[:data8.shape[0]]
X_test=X[data8.shape[0]:]
y_train=data8['label']
model=GradientBoostingRegressor()
model.fit(X_train, y_train)
print(model.score(X_train,y_train))
y_pre=model.predict(X_test)
data9['label']=y_pre
data9['label']=0.9*data9['label']+0.1*data9['跪舔']
data9.sort_values(by='label',ascending=False,inplace=True)
data9.to_csv('201709_score.csv')

#slice
data9=data9[['客户等级','客户名称','客户证件类型','用户状态','服务号码','用户在网时长（月）','入网时间','年龄','最近第1月收入(总账)(元)','本月累计通话时长(分钟)','月累计4G上网流量(KB)','本月累计2G上网流量(KB)','本月累计3G上网流量(KB)','租机计划','资费名称','合约低消（PPM）','资费档位（PPM）','是否4G手机用户标志(自注册)','4G卡用户标志','2G/3G/4G标志','label']]
import math
ds=[]
ds.append(data9[:6000].sort_values(by='客户名称',ascending=False))
ds.append(data9[6000:20000].sort_values(by='客户名称',ascending=False))
count= math.floor((data9.shape[0]-20001)/4)
for i in range(4):
    if i==3:
        ds.append(data9[20000+i*count:].sort_values(by='客户名称',ascending=False))
    else:
        ds.append(data9[20000+i*count:20000+(i+1)*count].sort_values(by='客户名称',ascending=False))
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