import pandas as pd
import util
from sklearn.cross_validation import train_test_split

def extract(data):
    featuresCols=['本月实际应收(总账)(元)', '最近第1月收入(总账)(元)',
       '最近第2月收入(总账)(元)', '最近第3月收入(总账)(元)', '最近第4月收入(总账)(元)', '最近第5月收入(总账)(元)',
       '本月通话时长(分钟)', '最近第2月通话时长(分钟)', '最近第3月通话时长(分钟)', '最近第4月通话时长(分钟)',
       '最近第5月通话时长(分钟)', '本月无线宽带总流量(KB)', '最近第1月无线宽带流量(KB)', '最近第2月无线宽带流量(KB)',
       '最近第3月无线宽带流量(KB)', '最近第4月无线宽带流量(KB)', '最近第5月无线宽带流量(KB)']
    import locale
    locale.setlocale(locale.LC_NUMERIC, '')
    for c in featuresCols:
        if data[c].dtype=='O':
            data[c]=data[c].apply(locale.atof)
    dataFeatureCon=data[featuresCols]
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import StandardScaler
    vec = DictVectorizer(sparse = False)
    X_dict = dataFeatureCon.T.to_dict().values()
    X_vec = vec.fit_transform(X_dict)
    X_vec = StandardScaler().fit_transform(X_vec)
    return X_vec

p1=pd.read_excel('套餐升档（分配数据）.xls',sheetname='第一批（1217）')
p2=pd.read_excel('套餐升档（分配数据）.xls',sheetname='第二批（3204）')
p3=pd.read_excel('套餐升档（分配数据）.xls',sheetname='第三批（11564）')
p4=pd.read_excel('套餐升档（分配数据）.xls',sheetname='第四批(48854)')
p=pd.concat([p1,p2,p3,p4],axis=0)
p=p[p['是否接通']=='是']
p['是否办理升档'].fillna('否',inplace='true')
a=pd.read_excel('D:/套餐升档all.xlsx')
data=pd.merge(p,a,how='left',on='服务号码')
data=data.drop_duplicates(['服务号码'])
data.loc[data['是否办理升档']=='是','是否办理升档']=1
data.loc[data['是否办理升档']=='否','是否办理升档']=0

#use GBDT group by agent
from sklearn.ensemble import GradientBoostingRegressor
grouped=data.groupby('外呼人员')
for n,g in grouped:
    if g.shape[0]<100:
        continue
    success=sum(g['是否办理升档']==1)
    rate=success*1.0/g.shape[0]
    if rate>0.3:
        print('{} {}({}/{})'.format(n,rate,success,g.shape[0]))
        X=extract(g)
        y=g['是否办理升档'].astype('int64')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model=GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pre=model.predict(X_test)
        util.showReport(y_pre,y_test,rate*100)
      

'''
#use GBDT and show all reports
X=extract(data)
y=data['是否办理升档'].astype('int64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import GradientBoostingRegressor
model=GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pre=model.predict(X_test)
util.showFigure1(y_pre,y_test)
util.showReport(y_pre,y_test,27)

#use RF and print feature_importances_
featuresCols=['本月实际应收(总账)(元)', '最近第1月收入(总账)(元)',
       '最近第2月收入(总账)(元)', '最近第3月收入(总账)(元)', '最近第4月收入(总账)(元)', '最近第5月收入(总账)(元)',
       '本月通话时长(分钟)', '最近第2月通话时长(分钟)', '最近第3月通话时长(分钟)', '最近第4月通话时长(分钟)',
       '最近第5月通话时长(分钟)', '本月无线宽带总流量(KB)', '最近第1月无线宽带流量(KB)', '最近第2月无线宽带流量(KB)',
       '最近第3月无线宽带流量(KB)', '最近第4月无线宽带流量(KB)', '最近第5月无线宽带流量(KB)']
import locale
locale.setlocale(locale.LC_NUMERIC, '')
for c in featuresCols:
    if data[c].dtype=='O':
        data[c]=data[c].apply(locale.atof)
X=data[featuresCols]
y=data['是否办理升档'].astype('int64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_jobs=-1)
model.fit(X_train, y_train)
y_pre=model.predict(X_test)
#util.showFigure1(y_pre,y_test)
util.showReport(y_pre,y_test,27)
print(model.feature_importances_)
'''



