import numpy as np
import pandas as pd
import locale
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report


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
    
def showFigure(a):
    x=[];y=[]
    for i in range(10,50,1):
        TP=sum(a[a['状态']=='1']['pre']>(i/100))
        FP=sum(a[a['状态']=='0']['pre']>(i/100))
        FN=sum(a[a['状态']=='1']['pre']<=(i/100))
        TN=sum(a[a['状态']=='0']['pre']<=(i/100))
        y.append((TP+TN)/(TP+FP+FN+TN))
        x.append(TP/(TP+FN))
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x,y,'ko--')
    ax.set_xlabel('Sample utilization of 1')
    ax.set_ylabel('Predictive Accuracy')
    
def showFigure1(y_pre,y_test):
    x=[];y=[]
    import pandas as pd
    a=pd.DataFrame({"pre":y_pre,"状态":y_test})
    for i in range(1,50,1):
        TP=sum(a[a['状态']==1]['pre']>(i/100))
        FP=sum(a[a['状态']==0]['pre']>(i/100))
        FN=sum(a[a['状态']==1]['pre']<=(i/100))
        TN=sum(a[a['状态']==0]['pre']<=(i/100))
        y.append((TP+TN)/(TP+FP+FN+TN))
        x.append(TP/(TP+FN))
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x,y,'ko--')
    ax.set_xlabel('Sample utilization of 1')
    ax.set_ylabel('Predictive Accuracy')
    
def showReport(y_pre,y_test,threshold):
    col='pre_'.join(str(threshold))
    a=pd.DataFrame({"pre":y_pre,"test":y_test})
    a[col]=0
    a.loc[a['pre']>(threshold/100),col]=1
    cm=confusion_matrix(y_test.astype('int64'),a[col])
    print(classification_report(y_test.astype('int64'),a[col]))
    plot_confusion_matrix(cm,[0,1])
    return a
          
          
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    