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

# Aula 2

## Regressão Linear

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

regressor = Sequential()

regressor.add(Dense(1, input_dim=1, kernel_initializer='Ones',
                    activation='linear',use_bias=False))

regressor.compile(loss='mean_squared_error',optimizer='adam')

regressor.summary()

regressor.fit(xtreino,ytreino)

y_predict= regressor.predict(xtreino) #Prevendo os dados de treino (o ajuste)

sns.lineplot(x=xtreino,y=ytreino,label='treino')
sns.lineplot(x=xtreino,y=y_predict[:,0],label='ajuste_treino')

d = {'tempo': xtreino, 'passageiros': y_predict[:,0]}
resultados = pd.DataFrame(data=d)

resultados

resultado_transf = sc.inverse_transform(resultados)

resultado_transf = pd.DataFrame(resultado_transf)
resultado_transf.columns = ['tempo','passageiros']

sns.lineplot(x="tempo",y="passageiros",data=passageiros)
sns.lineplot(x="tempo",y="passageiros",data=resultado_transf)

y_predict_teste= regressor.predict(xteste) #Prevendo os dados de teste(o futuro)

d = {'tempo': xteste, 'passageiros': y_predict_teste[:,0]}
resultados_teste = pd.DataFrame(data=d)

resultado_transf_teste = sc.inverse_transform(resultados_teste)

resultado_transf_teste = pd.DataFrame(resultado_transf_teste)
resultado_transf_teste.columns = ['tempo','passageiros']

sns.lineplot(x="tempo",y="passageiros",data=passageiros,label='dado_completo')
sns.lineplot(x="tempo",y="passageiros",data=resultado_transf,label='ajuste_treino')
sns.lineplot(x="tempo",y="passageiros",data=resultado_transf_teste,label='previsão')

## Regressão não-linear

regressor2 = Sequential()

regressor2.add(Dense(8, input_dim=1, kernel_initializer='random_uniform',
                     activation='sigmoid',use_bias=False))
regressor2.add(Dense(8, kernel_initializer='random_uniform',
                     activation='sigmoid',use_bias=False))
regressor2.add(Dense(1, kernel_initializer='random_uniform',
                     activation='linear',use_bias=False))

regressor2.compile(loss='mean_squared_error',optimizer='adam')
regressor2.summary()

regressor2.fit(xtreino,ytreino,epochs =500)

y_predict= regressor2.predict(xtreino) #Prevendo os dados de treino (o ajuste)

y_predict_teste= regressor2.predict(xteste) #Prevendo os dados de teste(o futuro)

sns.lineplot(x=xtreino,y=ytreino,label='treino')
sns.lineplot(x=xteste,y=yteste,label='teste')
sns.lineplot(x=xtreino,y=y_predict[:,0],label='ajuste_treino')
sns.lineplot(x=xteste,y=y_predict_teste[:,0],label='previsão')