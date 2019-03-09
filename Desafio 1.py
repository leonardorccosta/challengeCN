import pandas as pd
import numpy as np

dataset = pd.read_csv('train.csv')

nmt = dataset.loc[:,'NU_NOTA_MT']
nmt = nmt*3
nlc = dataset.loc[:,'NU_NOTA_LC']
nlc = nlc*1.5
ncn = dataset.loc[:,'NU_NOTA_CN']
ncn = ncn*2
nch = dataset.loc[:,'NU_NOTA_CH']
nch = nch*1
nred = dataset.loc[:,'NU_NOTA_REDACAO']
nred = nred*3

media = nred+nch+ncn+nlc+nmt
media = media/10.5



insc = dataset.loc[:,'NU_INSCRICAO']
b = pd.DataFrame(media, columns=['media'])
b['inscricao'] = insc
cols = b.columns.tolist()
cols = cols[-1:] + cols[:-1]
b = b[cols]
b = media.sort_values(ascending=False)

a= b.sort_values('media',ascending=False)


melhores = a.iloc[0:20]


##----------##----------##----------##----------##----------##----------##----------##----------

import requests
data ={
  "token": "insert_your_token_here",
  "email": "insert_your_email_here",
  "answer": [
    {"NU_INSCRICAO": "160000000001", "NOTA_FINAL": 623.3},
    {"NU_INSCRICAO": "160000000002", "NOTA_FINAL": 567.2},
    {"NU_INSCRICAO": "160000000003", "NOTA_FINAL": 403.1}
  ]
}
listed=[]
for i in range(len(melhores)):
    listed.append({"NU_INSCRICAO": melhores.inscricao.iloc[i], "NOTA_FINAL": melhores.media.iloc[i]})
data['answer'] = listed
r = requests.post('https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-1/submit', json=data)
print (r.content)
print (r.json)