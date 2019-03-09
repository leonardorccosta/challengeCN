### CHALLENGE 3 ######
import pandas as pd
import numpy as np

dataset1 = pd.read_csv('.../train.csv')
dataset3 = pd.read_csv('.../test3.csv')


#Preparar X de treino
col = dataset3.columns.tolist()

datasetcol = dataset1[col]
datasetcol2 = dataset1[col]
X=datasetcol
X = X.dropna(subset=['TX_RESPOSTAS_MT'])
for i in range(0,len(X.TX_RESPOSTAS_MT.iloc[2])-5,1):
    col = 'Q%i' % (i+1)
    X[col] = X.TX_RESPOSTAS_MT.str.slice(start=i, stop=i+1)
X = X.drop(columns='TX_RESPOSTAS_MT')
X = X.drop(columns='NU_INSCRICAO')
X.fillna(0, inplace=True)

#Preparar o Y de treino
y = dataset1[['TX_RESPOSTAS_MT']]
y = y.dropna(subset=['TX_RESPOSTAS_MT'])
for i in range(0,len(y.TX_RESPOSTAS_MT.iloc[2]),1):
    col = 'Q%i' % (i+1)
    y[col] = y.TX_RESPOSTAS_MT.str.slice(start=i, stop=i+1)
    
   
A = dataset1[['TX_RESPOSTAS_MT']]
A = A.dropna(subset=['TX_RESPOSTAS_MT'])
for i in range(0,len(A.TX_RESPOSTAS_MT.iloc[2]),1):
    col = 'Q%i' % (i+1)
    A[col] = A.TX_RESPOSTAS_MT.str.slice(start=i, stop=i+1)
y=y.iloc[:,-5:]

yd1 = pd.get_dummies(data=y)


#X  Entrega
Xentrega = dataset3.drop(columns='NU_INSCRICAO')
for i in range(0,len(Xentrega.TX_RESPOSTAS_MT.iloc[0]),1):
    col = 'Q%i' % (i+1)
    Xentrega[col] = Xentrega.TX_RESPOSTAS_MT.str.slice(start=i, stop=i+1)
Xentrega =Xentrega.drop(columns = 'TX_RESPOSTAS_MT')
Xentrega = pd.get_dummies(data=Xentrega)
Xentrega.fillna(0, inplace=True)
colentrega = Xentrega.columns.tolist()

#Dummies TREINO
Xt_dummies = pd.get_dummies(data=X)
Xtreino = Xt_dummies[colentrega]

#
#Xtreino['MEDIA3'] = (Xtreino['NU_NOTA_CN']+Xtreino['NU_NOTA_CH']+Xtreino['NU_NOTA_LC'])/3
#Xentrega['MEDIA3'] = (Xentrega['NU_NOTA_CN']+Xentrega['NU_NOTA_CH']+Xentrega['NU_NOTA_LC'])/3
#

#Separar o Dataset
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(Xt_dummies, yd1, test_size = 0, random_state = seed,max_features = 0.1)

#### RANDOM FOREST #####
from sklearn.ensemble import RandomForestClassifier

seed = 120
#Q41################
y = yd1.iloc[:,:7]
yr= np.reshape(np.argmax(y.values, axis=1),(len(y),1))

classifier = RandomForestClassifier(n_estimators=1500,criterion='entropy',n_jobs =-1,random_state = seed,max_features = 0.1)
classifier.fit(Xtreino,yr)

y_pred = classifier.predict(Xentrega)
#y_pred = np.argmax(y_pred, axis =1)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(A['Q41'])
q041 = list(encoder.inverse_transform(y_pred))
from sklearn.ensemble import RandomForestClassifier



#Q42################
y = yd1.iloc[:,7:14]

yr= np.reshape(np.argmax(y.values, axis=1),(len(y),1))

classifier = RandomForestClassifier(n_estimators=1500,criterion='entropy',n_jobs =-1,random_state =seed,max_features =0.1)
classifier.fit(Xtreino,yr)

y_pred = classifier.predict(Xentrega)
#y_pred = np.argmax(y_pred, axis =1)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(A['Q42'])
q042 = list(encoder.inverse_transform(y_pred))


#Q43############################
y = yd1.iloc[:,14:21]

yr= np.reshape(np.argmax(y.values, axis=1),(len(y),1))

classifier = RandomForestClassifier(n_estimators=1500,criterion='entropy',n_jobs =-1,random_state =seed,max_features = 0.1)
classifier.fit(Xtreino,yr)

y_pred = classifier.predict(Xentrega)
#y_pred = np.argmax(y_pred, axis =1)


y_encoded = encoder.fit_transform(A['Q43'])
q043 = list(encoder.inverse_transform(y_pred))


#Q44########################
y = yd1.iloc[:,21:28]
yr= np.reshape(np.argmax(y.values, axis=1),(len(y),1))
classifier = RandomForestClassifier(n_estimators=1500,criterion='entropy',n_jobs =-1,random_state = seed,max_features = 0.1)
classifier.fit(Xtreino,yr)

y_pred = classifier.predict(Xentrega)
#y_pred = np.argmax(y_pred, axis =1)

y_encoded = encoder.fit_transform(A['Q44'])
q044 = list(encoder.inverse_transform(y_pred))




#Q45######################
y = yd1.iloc[:,28:]
yr= np.reshape(np.argmax(y.values, axis=1),(len(y),1))

classifier = RandomForestClassifier(n_estimators=1500,criterion='entropy',n_jobs =-1,random_state = seed,max_features = 0.1)
classifier.fit(Xtreino,yr)

y_pred = classifier.predict(Xentrega)
#y_pred = np.argmax(y_pred, axis =1)


y_encoded = encoder.fit_transform(A['Q45'])
q045 = list(encoder.inverse_transform(y_pred))







######################################################################################
import requests
data={
  "token": "",
  "email": "",
  "answer": [
    {"NU_INSCRICAO": "160000000001", "TX_RESPOSTAS_MT": "ABACD"},
    {"NU_INSCRICAO": "160000000002", "TX_RESPOSTAS_MT": "DECAA"},
    {"NU_INSCRICAO": "160000000003", "TX_RESPOSTAS_MT": "DAADB"}
  ]
}

#dict copy
listed=[] #list
for i in range(len(dataset3)): #creation of list with dics for each candidate
    listed.append({"NU_INSCRICAO": dataset3.NU_INSCRICAO.iloc[i], "TX_RESPOSTAS_MT": q041[i]+q042[i]+q043[i]+q044[i]+q045[i]})
data['answer'] = listed


r = requests.post('https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-3/submit', json=data)
r.content