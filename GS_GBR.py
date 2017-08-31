import pandas as pd
import util
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

data5=pd.read_csv('D:/5月租机到期.csv',header = 0,encoding='gbk')
data5=util.preprocessing(data5)
X=features=util.extractFeatures(data5)
y=data5['状态']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

params = {'max_depth': [2,3,4,5], 'n_estimators':[100,300], 'learning_rate':[0.05,0.1,0.25,0.5]}
model=GradientBoostingRegressor()
gs = GridSearchCV(model, params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_estimator_)
print("Grid scores calculated on training set:")
for params, mean_score, scores in gs.grid_scores_:
    print("%0.3f for %r" % (mean_score, params))
print(gs.score(X_train,y_train))
print(gs.score(X_test,y_test))
joblib.dump(gs.best_estimator_,"D:/python/model/gbr_cv.pkl")
    
    
    
    
    
    
    
    
    
    
    
    
    
    