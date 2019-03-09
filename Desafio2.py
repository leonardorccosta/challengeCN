#Importar dataset
import pandas as pd
import numpy as np

dataset1 = pd.read_csv('.../train.csv')
dataset2 = pd.read_csv('.../test2.csv')


#Seperar as colunas que tem nos 2 datasets em comum
col = dataset2.columns.tolist()
treino = dataset1[col]
treino =treino.drop(columns='CO_UF_RESIDENCIA')
treino1 = treino.drop(columns=['CO_PROVA_CH','CO_PROVA_LC','CO_PROVA_CN'])

#Separar o Y
Y = dataset1[['NU_NOTA_MT']]
Y.fillna(0, inplace=True)


# Dummies X treino
Xtreino_d = pd.get_dummies(data=treino1)


# Preparo X envio
X3 = dataset2.drop(columns=['NU_INSCRICAO','CO_PROVA_CH','CO_PROVA_LC','CO_PROVA_CN'])
X_envio = pd.get_dummies(data=X3)
X_envio =X_envio.drop(columns='CO_UF_RESIDENCIA')
X_envio.fillna(0, inplace=True)
# Preparo X treino
colenvio = X_envio.columns.tolist()
X_treino=Xtreino_d[colenvio]
X_treino.fillna(0, inplace=True)



#Adicionar a coluna de Media das 4 outras materias
X_treino['MEDIA3'] = (X_treino['NU_NOTA_CN']+X_treino['NU_NOTA_CH']+X_treino['NU_NOTA_LC']+X_treino['NU_NOTA_REDACAO'])/4
X_envio['MEDIA3'] = (X_envio['NU_NOTA_CN']+X_envio['NU_NOTA_CH']+X_envio['NU_NOTA_LC']+X_envio['NU_NOTA_REDACAO'])/4



                                           #RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=850 ,criterion = 'mse', random_state=777, oob_score = True,n_jobs=-1, max_features=0.1)
regressor.fit(X_treino, Y)


#NOTES
#random = 7
#800 90.2776
#1000 90.2775
#850  90.2791
#900  90.2777
# 90.3480
# 90.3975 regressor = RandomForestRegressor(n_estimators=850 ,criterion = 'mae', random_state= 120, oob_score = True,n_jobs=-1, max_features=0.1)
#90.4107 regressor = RandomForestRegressor(n_estimators=850 ,criterion = 'mse', random_state=777, oob_score = True,n_jobs=-1, max_features=0.1)

#regressor = RandomForestRegressor(n_estimators= 850, random_state= 7, oob_score = True,n_jobs=-1, min_samples_split=2, max_features=0.1)


y_pred = regressor.predict(X_envio)
#y_pred[y_pred < 200] = 0
notamtm = list(y_pred)


                                              #ENVIO
import requests
data={
  "token": "",
  "email": "",
  "answer": [
    {"NU_INSCRICAO": "160000000001", "NU_NOTA_MT": 462.9},
    {"NU_INSCRICAO": "160000000002", "NU_NOTA_MT": 423.6},
    {"NU_INSCRICAO": "160000000003", "NU_NOTA_MT": 414.6}
  ]
}


#Substituição no dicionario

listed=[] #lista
for i in range(len(dataset2)): #loop candidato + nota respectiva
    listed.append({"NU_INSCRICAO": dataset2.NU_INSCRICAO.iloc[i], "NU_NOTA_MT": notamtm[i]})
data['answer'] = listed
                                          #REQUEST
r = requests.post('https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-2/submit', json=data)
r.content
lista = []
lista.insert(r.content)