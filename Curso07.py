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

with open("cbow_s300.txt") as f:
    for linha in range(30):
        print(next(f))

from gensim.models import KeyedVectors

modelo = KeyedVectors.load_word2vec_format("cbow_s300.txt")
modelo.most_similar("china")
modelo.most_similar("itália")
modelo.most_similar(positive=["brasil", "argentina"])

#nuvens -> nuvem : estrelas -> estrela
#nuvens + estrela - nuvem = estrelas

modelo.most_similar(positive=["nuvens", "estrela"], negative=["nuvem"])
modelo.most_similar(positive=["professor", "mulher"], negative=["homem"])