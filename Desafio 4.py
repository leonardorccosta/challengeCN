#Importar dataset
import pandas as pd
import numpy as np

dataset1 = pd.read_csv('.../train.csv')
dataset4 = pd.read_csv('.../test4.csv')

#Seperar as colunas que tem nos 2 datasets em comum
col = dataset4.columns.tolist()
treino = dataset1[col]
treino = treino.drop(columns=['NU_INSCRICAO'])

#Separar o Y
Y = dataset1[['IN_TREINEIRO']]
Y.fillna(0, inplace=True)


# Dummies X treino
Xtreino_d = pd.get_dummies(data=treino)


# Preparo X envio
X3 = dataset4.drop(columns=['NU_INSCRICAO'])
X_envio = pd.get_dummies(data=X3)
X_envio.fillna(0, inplace=True)
# Preparo X treino
colenvio = X_envio.columns.tolist()
X_treino=Xtreino_d[colenvio]
X_treino.fillna(0, inplace=True)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_treino, Y)

# Predicting the Test set results
y_pred = classifier.predict(X_envio)


y_pred = classifier.predict(X_envio)
dataenvio = pd.DataFrame(y_pred)
dataenvio = dataenvio.iloc[:,0].astype(str).values.tolist()

###################################################
import requests
data={
  "token": "",
  "email": "",
  "answer": [
    {"NU_INSCRICAO": "160000000001", "IN_TREINEIRO": "0"},
    {"NU_INSCRICAO": "160000000002", "IN_TREINEIRO": "1"},
    {"NU_INSCRICAO": "160000000003", "IN_TREINEIRO": "1"}
  ]
}

#dict copy
listed=[] #list
for i in range(len(dataset4)): #creation of list with dics for each candidate
    listed.append({"NU_INSCRICAO": dataset4.NU_INSCRICAO.iloc[i], "IN_TREINEIRO": dataenvio[i]})
data['answer'] = listed


r = requests.post('https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-4/submit', json=data)
r.content
