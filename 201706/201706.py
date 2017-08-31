import pandas as pd
import util
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
'''
#build 4/5 model
data5=pd.read_excel('201704.xlsx')
data5=util.preprocessing(data5)
X=features=util.extractFeatures(data5)
y=data5['状态']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model=GradientBoostingRegressor()
model.fit(X_train, y_train)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
y_pre=model.predict(X_test)
util.showFigure1(y_pre,y_test)
util.showReport(y_pre,y_test,20)
'''

#predict 6 with 5's model
data4=pd.read_excel('201705.xlsx')
data6=pd.read_excel('201706.xlsx')
data=pd.concat((data4,data6), axis=0, join='inner',ignore_index=True)
data=util.preprocessing(data)
X=features=util.extractFeatures(data)
y=data['状态']
X_train=X[:data4.shape[0]]
X_test=X[data4.shape[0]:]
y_train=y[:data4.shape[0]]
model=GradientBoostingRegressor()
model.fit(X_train, y_train)
print(model.score(X_train,y_train))
y_pre=model.predict(X_test)
data6['状态']=y_pre
data6.to_csv('201706_score.csv')
