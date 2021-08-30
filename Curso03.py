import pandas as pd

passageiros = pd.read_csv('Passageiros.csv')

passageiros.head()

import seaborn as sns

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 22

sns.lineplot(x='tempo',y='passageiros', data=passageiros,label='dado_completo')

## Escalando os dados

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(passageiros)

dado_escalado = sc.transform(passageiros)

x=dado_escalado[:,0] #Features - Características - Tempo
y=dado_escalado[:,1] #Alvo - Número de passageiros

import matplotlib.pyplot as plt

sns.lineplot(x=x,y=y,label='dado_escalado')
plt.ylabel('Passageiros')
plt.xlabel('Data')

## Dividindo em treino e teste

tamanho_treino = int(len(passageiros)*0.9) #Pegando 90% dos dados para treino
tamanho_teste = len(passageiros)-tamanho_treino #O resto vamos reservar para teste

xtreino = x[0:tamanho_treino]
ytreino = y[0:tamanho_treino]

xteste = x[tamanho_treino:len(passageiros)]
yteste = y[tamanho_treino:len(passageiros)]

sns.lineplot(x=xtreino,y=ytreino,label='treino')
sns.lineplot(x=xteste,y=yteste,label='teste')