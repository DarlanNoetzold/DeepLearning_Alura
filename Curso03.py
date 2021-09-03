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

# Aula 3

## Alterando a forma como passamos os dados

#Agora x e y vão valores diferentes. X vai conter o número de passageiros em um tempo anterior e y vai conter o número de passageiros em t+1, por exemplo.

vetor = pd.DataFrame(ytreino)[0]

import numpy as np

def separa_dados(vetor,n_passos):
  """Entrada: vetor: número de passageiros
               n_passos: número de passos no regressor
     Saída:
              X_novo: Array 2D
              y_novo: Array 1D - Nosso alvo
  """
  X_novo, y_novo = [], []
  for i in range(n_passos,vetor.shape[0]):
    X_novo.append(list(vetor.loc[i-n_passos:i-1]))
    y_novo.append(vetor.loc[i])
  X_novo, y_novo = np.array(X_novo), np.array(y_novo)
  return X_novo, y_novo

xtreino_novo, ytreino_novo = separa_dados(vetor,1)

print(xtreino_novo[0:5]) #X

print(ytreino_novo[0:5]) #y

## Agora vamos separar o teste

vetor2 = pd.DataFrame(yteste)[0]

xteste_novo, yteste_novo = separa_dados(vetor2,1)

## Voltando para as redes neurais

regressor3 = Sequential()

regressor3.add(Dense(8, input_dim=1, kernel_initializer='ones', activation='linear',use_bias=False))
regressor3.add(Dense(64, kernel_initializer='random_uniform', activation='sigmoid',use_bias=False))
regressor3.add(Dense(1, kernel_initializer='random_uniform', activation='linear',use_bias=False))
regressor3.compile(loss='mean_squared_error',optimizer='adam')
regressor3.summary()

regressor3.fit(xtreino_novo,ytreino_novo,epochs =100)

y_predict_novo = regressor3.predict(xtreino_novo)

sns.lineplot(x='tempo',y=ytreino_novo,data=passageiros[1:129],label='treino')
sns.lineplot(x='tempo',y=pd.DataFrame(y_predict_novo)[0],data=passageiros[1:129],label='ajuste_treino')

y_predict_teste_novo = regressor3.predict(xteste_novo)

resultado = pd.DataFrame(y_predict_teste_novo)[0]

sns.lineplot(x='tempo',y=ytreino_novo,data=passageiros[1:129],label='treino')
sns.lineplot(x='tempo',y=pd.DataFrame(y_predict_novo)[0],data=passageiros[1:129],label='ajuste_treino')

sns.lineplot(x='tempo',y=yteste_novo,data=passageiros[130:144],label='teste')
sns.lineplot(x='tempo',y=resultado.values,data=passageiros[130:144],label='previsão')

## Janelas

xtreino_novo, ytreino_novo = separa_dados(vetor,4)

xtreino_novo[0:5] #X

ytreino_novo[0:5] #y

xteste_novo, yteste_novo = separa_dados(vetor2,4)

regressor4 = Sequential()

regressor4.add(Dense(8, input_dim=4, kernel_initializer='random_uniform', activation='linear',use_bias=False)) #relu
regressor4.add(Dense(64, kernel_initializer='random_uniform', activation='sigmoid',use_bias=False)) #relu
regressor4.add(Dense(1, kernel_initializer='random_uniform', activation='linear',use_bias=False))
regressor4.compile(loss='mean_squared_error',optimizer='adam')
regressor4.summary()

regressor4.fit(xtreino_novo,ytreino_novo,epochs =300)

y_predict_teste_novo = regressor4.predict(xteste_novo)

resultado = pd.DataFrame(y_predict_teste_novo)[0]

sns.lineplot(x='tempo',y=ytreino_novo,data=passageiros[4:129],label='treino')
sns.lineplot(x='tempo',y=pd.DataFrame(y_predict_novo)[0],data=passageiros[4:129],label='ajuste_treino')

sns.lineplot(x='tempo',y=yteste_novo,data=passageiros[133:144],label='teste')
sns.lineplot(x='tempo',y=resultado.values,data=passageiros[133:144],label='previsão')

#Nova base de dados

bike = pd.read_csv('bicicletas.csv')

bike.head()

bike['datas'] = pd.to_datetime(bike['datas'])

sns.lineplot(x='datas',y='contagem', data=bike)
plt.xticks(rotation=70)

## Escalando os dados

sc2 = StandardScaler()

sc2.fit(bike['contagem'].values.reshape(-1,1))

y = sc2.transform(bike['contagem'].values.reshape(-1,1))

## Dividindo em treino e teste

tamanho_treino = int(len(bike)*0.9) #Pegando 90% dos dados para treino
tamanho_teste = len(bike)-tamanho_treino #O resto vamos reservar para teste

ytreino = y[0:tamanho_treino]
yteste = y[tamanho_treino:len(bike)]

sns.lineplot(x='datas',y=ytreino[:,0],data=bike[0:tamanho_treino],label='treino')
sns.lineplot(x='datas',y=yteste[:,0], data=bike[tamanho_treino:len(bike)],label='teste')
plt.xticks(rotation=70)

vetor = pd.DataFrame(ytreino)[0]

xtreino_novo, ytreino_novo = separa_dados(vetor,10)

print(xtreino_novo[0:5])

print(ytreino_novo[0:5])

vetor2 = pd.DataFrame(yteste)[0]

xteste_novo, yteste_novo = separa_dados(vetor2,10)

## O que a LSTM espera

# A entrada de redes recorrentes deve possuir a seguinte forma para a entrada (número de amostras, número de passos no tempo,
# e número de atributos por passo no tempo).

xtreino_novo = xtreino_novo.reshape((xtreino_novo.shape[0],xtreino_novo.shape[1],1))

print(xtreino_novo.shape)

xteste_novo = xteste_novo.reshape((xteste_novo.shape[0],xteste_novo.shape[1],1))

## Usando a LSTM

from tensorflow.keras.layers import LSTM

recorrente = Sequential()

recorrente.add(LSTM(128, input_shape=(xtreino_novo.shape[1],xtreino_novo.shape[2])
                    ))
recorrente.add(Dense(units=1))

recorrente.compile(loss='mean_squared_error',optimizer='RMSProp')
recorrente.summary()

resultado = recorrente.fit(xtreino_novo,ytreino_novo,validation_data=(xteste_novo,yteste_novo),epochs=100)

y_ajustado = recorrente.predict(xtreino_novo)

sns.lineplot(x='datas',y=ytreino[:,0],data=bike[0:tamanho_treino],label='treino')
sns.lineplot(x='datas',y=y_ajustado[:,0],data=bike[0:15662],label='ajuste_treino')
plt.xticks(rotation=70)

y_predito = recorrente.predict(xteste_novo)

sns.lineplot(x='datas',y=yteste[:,0], data=bike[tamanho_treino:len(bike)],label='teste')
sns.lineplot(x='datas',y=y_predito[:,0], data=bike[tamanho_treino+10:len(bike)],marker='.',label='previsão')
plt.xticks(rotation=70)

from tensorflow.keras.layers import GRU

recorrente_g = Sequential()

recorrente_g.add(GRU(128, input_shape=(xtreino_novo.shape[1],xtreino_novo.shape[2])
                    ))
recorrente_g.add(Dense(units=1))

recorrente_g.compile(loss='mean_squared_error',optimizer='RMSProp')
recorrente_g.summary()

resultado2 = recorrente_g.fit(xtreino_novo,ytreino_novo,
                              validation_data=(xteste_novo,yteste_novo),epochs=100)

y_predito2 = recorrente_g.predict(xteste_novo)

sns.lineplot(x='datas',y=yteste[:,0], data=bike[tamanho_treino:len(bike)])
sns.lineplot(x='datas',y=y_predito2[:,0], data=bike[tamanho_treino+10:len(bike)],marker='.')
plt.legend(['conhecido','estimado'])
plt.xticks(rotation=70)

print(resultado2.history.keys())

plt.plot(resultado.history['loss'])
plt.plot(resultado.history['val_loss'])
plt.legend(['treino','teste'])

plt.plot(resultado2.history['loss'])
plt.plot(resultado2.history['val_loss'])
plt.legend(['treino','teste'])
