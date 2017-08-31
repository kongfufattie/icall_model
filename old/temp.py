# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
data=pd.read_csv('D:/5月租机到期.csv',header = 0,encoding='gbk')
#删除列['客户名称','客户状态','入网时间','用户状态','分配日期','客户名称','服务号码','主产品细类','客户证件类型','个人所属政企客户名称','用户编号','组合销售品成员角色','主产品细类二级','离网概率','合约到期续约概率']
data=data.drop(['客户名称','客户状态','入网时间','用户状态','分配日期','客户名称','服务号码','主产品细类','客户证件类型',
                '个人所属政企客户名称','用户编号','组合销售品成员角色','主产品细类二级','离网概率','合约到期续约概率'],axis=1)

data.loc[data['终端出现次序']!=1,'终端出现次序']=0
data['资费档位（PPM）'].fillna(0,inplace='true')
data['合约低消（PPM）'].fillna(0,inplace='true')
data['协议在网月份数（PPM）'].fillna(0,inplace='true')
data['是否翼支付用户'].fillna('否',inplace='true')
print(data.head())

data=data[data['状态']!='未处理']

#处理连续值特征
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)
featueConCols=['用户在网时长（月）', '本月实收总费用(元)', '本月通话时长(分钟)', '本月短信条数(条)', '月累计4G上网流量(KB)',
               '本月2G上网流量(KB)', '本月3G上网流量(KB)', '微信流量(KB)','年龄','易信流量(KB)','用户信用度', '可兑积分(分)']
dataFeatureCon=data[featueConCols]
X_dictCon = dataFeatureCon.T.to_dict().values()
X_vec_con = vec.fit_transform(X_dictCon)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_vec_con)
X_vec_con = scaler.transform(X_vec_con)
print(X_vec_con)
#处理离散特征，也可试试preprocessing.LabelEncoder()
'''
featureCatCols=['租机计划', '资费名称', '手机型号','4G卡用户标志', '标志_维挽拍照用户', '用户发展二级部门', '客户等级',
                '子品牌', '品牌', '资费档位（PPM）', '合约低消（PPM）','协议在网月份数（PPM）', '手机品牌', 
                '终端出现次序', '捆绑机型', '标志_预付后付','是否4G手机用户标志(自注册)', '本月4G终端且开卡']
dataFeatureCat=data[featureCatCols]
X_dictCat = dataFeatureCat.T.to_dict().values()
X_vec_cat = vec.fit_transform(X_dictCat)
enc = preprocessing.OneHotEncoder()
enc.fit(X_vec_cat)
X_vec_cat = enc.transform(X_vec_cat).toarray()
print(X_vec_cat)
'''
#训练GBDT
import time
t1=time.time()
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier()
gbdt.fit(X_vec_con[:30000,:],data.iloc[:30000]['状态'])
#iloc是将序列当作数组来访问，下标会从0开始
t2=time.time()
print(t2-t1)
#保存训练模型,joblib.load可恢复
from sklearn.externals import joblib
joblib.dump(gbdt, "D:/python/model/icall_train_model.pkl")
predictLabels=gbdt.predict(X_vec_con[30000:,:])
t3=time.time()
print(t3-t2)
print(predictLabels)
testLabels=data.iloc[30000:]['状态']
sum(predictLabels==testLabels)










