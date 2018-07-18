# the code for perprocess data and perpare features

import numpy as np
import pandas as pd

#读入数据
train_feature=pd.read_csv('train.tsv',sep='\t')
train_id=pd.read_csv('train_id.tsv',sep='\t')


#处理  FTR51
medicine_num=np.array([ train_feature['FTR51'][i].count(',')+1 for i in np.arange(train_feature['FTR51'].shape[0])])

train_feature['FTR51求和']=medicine_num


#合并两个 dataframe,示例标记：异常 +1，正常 0
train_merge = pd.merge(train_feature,train_id,on='PERSONID')


#计算每个人的条目数,增加一列
train_feature['条目数']=1

apply_num=train_feature.groupby('PERSONID') #把personID 作为了行索引，少了一列

#对同一个人的所有条目都求和
train_person1=apply_num.sum()   #(15000 52) 
#自动把非数值的特征都自动丢了

train_person1.describe()

#对同一个人的所有条目求平均,但 apply_num 只求和
tmp=apply_num['条目数'].sum()
train_person=apply_num.max()
train_person['条目数']=tmp

#特征的特性
print('对应到人的特征特性:',train_person.describe())

#计算对应于人的条目个数

sum_column_person =np.array([np.sum(train_person.iloc[:,i]) 
                        for i in np.arange(0,52)])

#计算每一个数值特征不为0的个数
sum_column_nozero=np.array([np.sum(train_person.iloc[:,i]!=0) 
                        for i in np.arange(0,52)])

#去掉一些 Feature FTR6、FTR11、FTR13、FTR19、FTR22、FTR24
train_feature=train_feature.drop(['FTR6','FTR11','FTR13','FTR19','FTR22','FTR24'],axis=1)  
#(1369=8146,50)

#多了PERSONID， APPLYNO，FTR51、时间戳

train_person=train_person.drop(['FTR6','FTR11','FTR13','FTR19','FTR22','FTR24'],axis=1)
#（15000,46)


X_train=train_feature.iloc[:,2:47]    #读出 去除了上述feature后的 FTR0-FTR50
X_train.shape  #（1368146,45)  45=51-6


#交易条目异常标记
y_apply_train=train_merge['LABEL']
np.sum(y_apply_train)  #130399个 1，将近 10%


#from sklearn.model_selection import StratifiedKFold
#skf = StratifiedKFold(n_splits=5)
#skf.get_n_splits(X_train, y_train)
#skf.split(X_train, y_train)

X_person_train=train_person  #(15000,46) 多了一个 applynum
#y_person_train=train_id['LABEL']
#检查是否对齐
