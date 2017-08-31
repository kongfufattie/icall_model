import numpy as np
import pandas as pd
import locale
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report


def preprocessing(data):
    #易信流量(KB)中绝大多数为0，故转为离散特征“是否有易信流量”
    data['资费档位（PPM）'].fillna(0,inplace='true')
    data['合约低消（PPM）'].fillna(0,inplace='true')
    data['手机型号ID'].fillna('无',inplace='true')
    data['手机品牌'].fillna('无',inplace='true')
    data['捆绑机型'].fillna('无',inplace='true')
    locale.setlocale(locale.LC_NUMERIC, '')
    if data['本月累计短信条数(次)'].dtype!='int64':
        data['本月累计短信条数(次)']=data['本月累计短信条数(次)'].apply(locale.atof)
    print(data.shape)
    return data

def extractFeatures(data):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn import preprocessing
    vec = DictVectorizer(sparse = False)
    #miss '用户在网时长（月）'
    featueConCols=[ '最近第1月收入(总账)(元)', '最近第1月通话时长(分钟)', '本月累计短信条数(次)', '月累计4G上网流量(KB)',
                   '本月累计2G上网流量(KB)', '本月累计3G上网流量(KB)','年龄']
    dataFeatureCon=data[featueConCols]
    #X_dictCon = dataFeatureCon.T.to_dict().values()
    X_dictCon = [dict(r.iteritems()) for _, r in dataFeatureCon.iterrows()]
    X_vec_con = vec.fit_transform(X_dictCon)
    scaler = preprocessing.StandardScaler().fit(X_vec_con)
    X_vec_con = scaler.transform(X_vec_con)
    print(X_vec_con.shape)
    #处理离散特征，也可试试preprocessing.LabelEncoder()
    featureCatCols=['租机计划', '资费名称', '手机型号ID','4G卡用户标志', '标志_维挽拍照用户', '客户等级',
                    '子品牌', '品牌', '资费档位（PPM）', '合约低消（PPM）', '手机品牌', 
                    '捆绑机型', '标志_预付后付','是否4G手机用户标志(自注册)', '4G终端且开卡用户标志']
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
    featueConCols=['最近第1月收入(总账)(元)', '最近第1月通话时长(分钟)', '本月累计短信条数(次)', '月累计4G上网流量(KB)',
                   '本月累计2G上网流量(KB)', '本月累计3G上网流量(KB)','年龄']
    dataFeatureCon=data[featueConCols]
    #X_dictCon = dataFeatureCon.T.to_dict().values()
    X_dictCon = [dict(r.iteritems()) for _, r in dataFeatureCon.iterrows()]
    X_vec_con = vec.fit_transform(X_dictCon)
    scaler = preprocessing.StandardScaler().fit(X_vec_con)
    X_vec_con = scaler.transform(X_vec_con)
    print(X_vec_con)
    return X_vec_con

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
    
def showPRfigure(y_pre,y_test):
    x=[];y=[]
    import pandas as pd
    a=pd.DataFrame({"pre":y_pre,"状态":y_test})
    for i in range(1,50,1):
        TP=sum(a[a['状态']==1]['pre']>(i/100))
        FP=sum(a[a['状态']==0]['pre']>(i/100))
        FN=sum(a[a['状态']==1]['pre']<=(i/100))
        y.append(TP/(TP+FP))
        x.append(TP/(TP+FN))
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x,y,'ko--')
    ax.set_xlabel('recall of 1')
    ax.set_ylabel('precsion of 1')
    
def showReport(y_pre,y_test,threshold):
    col='pre_'.join(str(threshold))
    a=pd.DataFrame({"pre":y_pre,"test":y_test})
    a[col]=0
    a.loc[a['pre']>(threshold/100),col]=1
    cm=confusion_matrix(y_test.astype('int64'),a[col])
    print(classification_report(y_test.astype('int64'),a[col]))
    plot_confusion_matrix(cm,[0,1])
    return a
          
          
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    