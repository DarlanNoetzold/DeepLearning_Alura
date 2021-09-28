import pandas as pd
artigo_treino = pd.read_csv("treino.csv")
artigo_teste = pd.read_csv("teste.csv")
artigo_treino.sample(5)

artigo_teste.sample(5)
print(artigo_teste.iloc[643].title)

from sklearn.feature_extraction.text import CountVectorizer

texto = [
    "tenha um bom dia",
    "tenha um péssimo dia",
    "tenha um ótimo dia",
    "tenha um dia ruim python java alura caelum papa"

]

vetorizador = CountVectorizer()
vetorizador.fit(texto)
print(vetorizador.vocabulary_)

print(vetorizador.vocabulary_)

vetor_bom = vetorizador.transform(["bom"])
print(vetor_bom.toarray())

vetor_otimo = vetorizador.transform(["ótimo"])
print(vetor_otimo.toarray())

