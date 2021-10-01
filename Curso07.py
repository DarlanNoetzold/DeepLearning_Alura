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

artigo_treino.title.loc[12]
import string
import nltk
nltk.download('punkt')

def tokenizador(texto):
    texto = texto.lower()
    lista_alfanumerico = []

    for token_valido in nltk.word_tokenize(texto):
        if token_valido in string.punctuation: continue
        lista_alfanumerico.append(token_valido)

    return lista_alfanumerico

tokenizador("Texto Exemplo, 1234.")

import numpy as np
def combinacao_de_vetores_por_soma(palavras_numeros):
    vetor_resultante = np.zeros(300)
    for pn in palavras_numeros:
        try:
            vetor_resultante += modelo.get_vector(pn)
        except KeyError:
            if pn.isnumeric():
               pn = "0"*len(pn)
               vetor_resultante += modelo.get_vector(pn)
            else:
                vetor_resultante += modelo.get_vector("unknown")

    return vetor_resultante

palavras_numeros = tokenizador("texto exemplo caelumx")
vetor_texto = combinacao_de_vetores_por_soma(palavras_numeros)
print(len(vetor_texto))
print(vetor_texto)