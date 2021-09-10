import pandas as pd

resenha = pd.read_csv("imdb-reviews-pt-br.csv")
print(resenha.head())

from sklearn.model_selection import train_test_split

treino, teste, classe_treino, classe_teste = train_test_split(resenha.text_pt,
                                                              resenha.sentiment,
                                                              random_state = 42)

from sklearn.linear_model import LogisticRegression

regressao_logistica = LogisticRegression()
regressao_logistica.fit(treino, classe_treino)
acuracia = regressao_logistica.score(teste, classe_teste)
print(acuracia)

print("Negativa \n")
print(resenha.text_pt[189])

print("Positivo \n")
print(resenha.text_pt[49002])

print(resenha.sentiment.value_counts())
print(resenha.head())

classificacao = resenha["sentiment"].replace(["neg", "pos"], [0,1])
resenha["classificacao"] = classificacao

print(resenha.tail())
