from sklearn.externals import joblib
import pandas as pd
import numpy as np
import locale
import matplotlib.pyplot as plt

def preprocessing(d):
    data=d[['用户在网时长（月）', '本月实收总费用(元)', '本月通话时长(分钟)', '本月短信条数(条)', '月累计4G上网流量(KB)',
               '本月2G上网流量(KB)', '本月3G上网流量(KB)', '微信流量(KB)','年龄','易信流量(KB)','用户信用度', '可兑积分(分)',
                '租机计划', '资费名称', '手机型号','4G卡用户标志', '标志_维挽拍照用户', '用户发展二级部门', '客户等级',
                '子品牌', '品牌', '资费档位（PPM）', '合约低消（PPM）','协议在网月份数（PPM）', '手机品牌', '是否翼支付用户',
                '终端出现次序', '捆绑机型', '标志_预付后付','是否4G手机用户标志(自注册)', '本月4G终端且开卡',
                '状态']]
    #易信流量(KB)中绝大多数为0，故转为离散特征“是否有易信流量”
    data.loc[data['易信流量(KB)']!=0,'易信流量(KB)']=1
    data.loc[data['终端出现次序']!=1,'终端出现次序']=0
    data['资费档位（PPM）'].fillna(0,inplace='true')
    data['合约低消（PPM）'].fillna(0,inplace='true')
    data['协议在网月份数（PPM）'].fillna(0,inplace='true')
    data['是否翼支付用户'].fillna('否',inplace='true')
    data['手机型号'].fillna('无',inplace='true')
    data['用户发展二级部门'].fillna('无',inplace='true')
    data['手机品牌'].fillna('无',inplace='true')
    data['捆绑机型'].fillna('无',inplace='true')
    locale.setlocale(locale.LC_NUMERIC, '')
    if data['可兑积分(分)'].dtype!='int64':
        data['可兑积分(分)']=data['可兑积分(分)'].apply(locale.atof)
    if data['微信流量(KB)'].dtype!='int64':
        data['微信流量(KB)']=data['微信流量(KB)'].apply(locale.atof)
    if data['本月短信条数(条)'].dtype!='int64':
        data['本月短信条数(条)']=data['本月短信条数(条)'].apply(locale.atof)
    data=data[data['状态']!='未处理']
    print(data.shape)
    return data

def extractFeatures(data):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn import preprocessing
    vec = DictVectorizer(sparse = False)
    featueConCols=['用户在网时长（月）', '本月实收总费用(元)', '本月通话时长(分钟)', '本月短信条数(条)', '月累计4G上网流量(KB)',
                   '本月2G上网流量(KB)', '本月3G上网流量(KB)', '微信流量(KB)','年龄','用户信用度', '可兑积分(分)']
    dataFeatureCon=data[featueConCols]
    #X_dictCon = dataFeatureCon.T.to_dict().values()
    X_dictCon = [dict(r.iteritems()) for _, r in dataFeatureCon.iterrows()]
    X_vec_con = vec.fit_transform(X_dictCon)
    scaler = preprocessing.StandardScaler().fit(X_vec_con)
    X_vec_con = scaler.transform(X_vec_con)
    print(X_vec_con.shape)
    #处理离散特征，也可试试preprocessing.LabelEncoder()
    featureCatCols=['租机计划', '资费名称', '手机型号','4G卡用户标志', '标志_维挽拍照用户', '用户发展二级部门', '客户等级',
                    '子品牌', '品牌', '资费档位（PPM）', '合约低消（PPM）','协议在网月份数（PPM）', '手机品牌','是否翼支付用户', 
                    '终端出现次序', '捆绑机型', '标志_预付后付','是否4G手机用户标志(自注册)', '本月4G终端且开卡','易信流量(KB)']
    dataFeatureCat=data[featureCatCols]
    X_dictCat = dataFeatureCat.T.to_dict().values()
    X_vec_cat = vec.fit_transform(X_dictCat)
    enc = preprocessing.OneHotEncoder()
    enc.fit(X_vec_cat)
    X_vec_cat = enc.transform(X_vec_cat).toarray()
    print(X_vec_cat.shape)
    return np.concatenate((X_vec_con,X_vec_cat), axis=1)
    
def extractConFeatures(data):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn import preprocessing
    vec = DictVectorizer(sparse = False)
    featueConCols=['用户在网时长（月）', '本月实收总费用(元)', '本月通话时长(分钟)', '本月短信条数(条)', '月累计4G上网流量(KB)',
                   '本月2G上网流量(KB)', '本月3G上网流量(KB)', '微信流量(KB)','年龄','用户信用度', '可兑积分(分)']
    dataFeatureCon=data[featueConCols]
    #X_dictCon = dataFeatureCon.T.to_dict().values()
    X_dictCon = [dict(r.iteritems()) for _, r in dataFeatureCon.iterrows()]
    X_vec_con = vec.fit_transform(X_dictCon)
    scaler = preprocessing.StandardScaler().fit(X_vec_con)
    X_vec_con = scaler.transform(X_vec_con)
    print(X_vec_con)
    return X_vec_con
    
def extractCatFeatures(data):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn import preprocessing
    vec = DictVectorizer(sparse = False)
    featureCatCols=['租机计划', '资费名称', '手机型号','4G卡用户标志', '标志_维挽拍照用户', '用户发展二级部门', '客户等级',
                    '子品牌', '品牌', '资费档位（PPM）', '合约低消（PPM）','协议在网月份数（PPM）', '手机品牌','是否翼支付用户', 
                    '终端出现次序', '捆绑机型', '标志_预付后付','是否4G手机用户标志(自注册)', '本月4G终端且开卡','易信流量(KB)']
    dataFeatureCat=data[featureCatCols]
    X_dictCat = dataFeatureCat.T.to_dict().values()
    X_vec_cat = vec.fit_transform(X_dictCat)
    enc = preprocessing.OneHotEncoder()
    enc.fit(X_vec_cat)
    X_vec_cat = enc.transform(X_vec_cat).toarray()
    print(X_vec_cat)
    return X_vec_cat

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

    
def plot_pr_curve(recall, precision):
    plt.clf()
    plt.plot(recall, precision, lw=1, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.show()
    
data=pd.read_csv('D:/5月租机到期.csv',header = 0,encoding='gbk')
print(data.shape)
data=preprocessing(data)
X=extractFeatures(data)
y=data['状态']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#base score
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
#cm and report
from sklearn.metrics import confusion_matrix,classification_report
y_pre=clf.predict(X_test)
print(confusion_matrix(y_test,y_pre))
print(classification_report(y_test,y_pre))
#p/r curve and pr_score(auc) of '1'.(each class has its own curve)
from sklearn.metrics import precision_recall_curve,auc
proba=clf.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, proba[:,1],pos_label='1')
pr_score=auc(recall,precision)
print(pr_score)
plot_pr_curve(recall,precision)


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
y_pre=clf.predict(X_test)
cm=confusion_matrix(y_test,y_pre)
print(classification_report(y_test,y_pre))
plot_confusion_matrix(cm,[0,1])

from sklearn.grid_search import GridSearchCV
params={"max_depth":[10,20,30,40]}
dt=DecisionTreeClassifier()
#gridsearch scoring='recall' if really concern about positive class
gs=GridSearchCV(dt,params,cv=3,scoring='f1',n_jobs=-1)
gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)

#ensemble
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
y_pre=clf.predict(X_test)
print(classification_report(y_test,y_pre))
#from xgboost import XGBClassifier
#clf=XGBClassifier()
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
y_pre=clf.predict(X_test)
print(classification_report(y_test,y_pre))
proba=clf.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, proba[:,1],pos_label='1')
pr_score=auc(recall,precision)
print(pr_score)
plot_pr_curve(recall,precision)

#rf with cv gs.best_params_{'max_depth': 1, 'min_samples_leaf': 1, 'min_samples_split': 1}
params = {'max_depth': [1,2,3,4,5], 'min_samples_split': [1,2,3], 'min_samples_leaf': [1,2,3]}
rf=RandomForestClassifier(n_estimators=10,max_features='auto', 
            bootstrap=True, oob_score=False, n_jobs=-1, 
            random_state=None, verbose=0)
gs = GridSearchCV(rf, params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_estimator_)
print("Grid scores calculated on training set:")
for params, mean_score, scores in gs.grid_scores_:
    print("%0.3f for %r" % (mean_score, params))
print(gs.score(X_train,y_train))
print(gs.score(X_test,y_test))
y_pre=gs.predict(X_test)
print(classification_report(y_test,y_pre))
proba=gs.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, proba[:,1],pos_label='1')
pr_score=auc(recall,precision)
print(pr_score)
plot_pr_curve(recall,precision)


#test util
import util
data5=pd.read_csv('D:/5月租机到期.csv',header = 0,encoding='gbk')
data5=preprocessing(data5)
X=features=extractFeatures(data5)
y=data5['状态']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model=joblib.load('D:/python/model/GBR_5.pkl')
y_pre=model.predict(X_test)
util.showFigure1(y_pre,y_test)
util.showReport(y_pre,y_test,20)











