import pandas as pd
import spacy

dados_treino = pd.read_csv("/content/drive/My Drive/curso_word2vec/treino.csv")
dados_treino.sample(5)

#!python -m spacy download pt_core_news_sm

nlp = spacy.load("pt_core_news_sm")

texto = "Rio de Janeiro é uma cidade maravilhosa"
doc = nlp(texto)
textos_para_tratamento = (titulos.lower() for titulos in dados_treino["title"])

def trata_textos(doc):
    tokens_validos = []
    for token in doc:
        e_valido = not token.is_stop and token.is_alpha
        if e_valido:
            tokens_validos.append(token.text)

    if len(tokens_validos) > 2:
        return  " ".join(tokens_validos)

texto = "Rio de Janeiro 1231231 ***** @#$ é uma cidade maravilhosa!"
doc = nlp(texto)
trata_textos(doc)

from time import time

t0 = time()
textos_tratados = [trata_textos(doc) for doc in nlp.pipe(textos_para_tratamento,
                                                        batch_size = 1000,
                                                        n_process = -1)]

tf = time() - t0
print(tf/60)

titulos_tratados = pd.DataFrame({"titulo": textos_tratados})
titulos_tratados.head()


from gensim.models import Word2Vec

w2v_modelo = Word2Vec(sg = 0,
                      window = 2,
                      size = 300,
                      min_count = 5,
                      alpha = 0.03,
                      min_alpha = 0.007)

print(len(titulos_tratados))
titulos_tratados = titulos_tratados.dropna().drop_duplicates()
print(len(titulos_tratados))

lista_lista_tokens = [titulo.split(" ") for titulo in titulos_tratados.titulo]


import logging
logging.basicConfig(format="%(asctime)s : - %(message)s", level = logging.INFO)
w2v_modelo = Word2Vec(sg = 0,
                      window = 2,
                      size = 300,
                      min_count = 5,
                      alpha = 0.03,
                      min_alpha = 0.007)
w2v_modelo.build_vocab(lista_lista_tokens, progress_per=5000)

w2v_modelo.train(lista_lista_tokens,
                 total_examples=w2v_modelo.corpus_count,
                 epochs = 30)
w2v_modelo.wv.most_similar("google")


#Treinamento do modelo Skip-Gram
w2v_modelo_sg = Word2Vec(sg = 1,
                      window = 5,
                      size = 300,
                      min_count = 5,
                      alpha = 0.03,
                      min_alpha = 0.007)

w2v_modelo_sg.build_vocab(lista_lista_tokens, progress_per=5000)

w2v_modelo_sg.train(lista_lista_tokens,
                 total_examples=w2v_modelo_sg.corpus_count,
                 epochs = 30)

w2v_modelo_sg.wv.most_similar("google")

##################################################################################################
#!python -m spacy download pt_core_news_sm      --rodar no colab

import spacy
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

w2v_modelo_cbow = KeyedVectors.load_word2vec_format("/content/drive/My Drive/curso_word2vec/modelo_cbow.txt")
w2v_modelo_sg = KeyedVectors.load_word2vec_format("/content/drive/My Drive/curso_word2vec/modelo_skipgram.txt")
artigo_treino = pd.read_csv("/content/drive/My Drive/curso_word2vec/treino.csv")
artigo_teste = pd.read_csv("/content/drive/My Drive/curso_word2vec/teste.csv")

nlp = spacy.load("pt_core_news_sm", disable=["paser", "ner", "tagger", "textcat"])

def tokenizador(texto):
    doc = nlp(texto)
    tokens_validos = []
    for token in doc:
        e_valido = not token.is_stop and token.is_alpha
        if e_valido:
            tokens_validos.append(token.text.lower())

    return tokens_validos


texto = "Rio de Janeiro 1231231 ***** @#$ é uma cidade maravilhosa!"
tokens = tokenizador(texto)
print(tokens)


def combinacao_de_vetores_por_soma(palavras, modelo):
    vetor_resultante = np.zeros((1, 300))

    for pn in palavras:
        try:
            vetor_resultante += modelo.get_vector(pn)

        except KeyError:
            pass

    return vetor_resultante


vetor_texto = combinacao_de_vetores_por_soma(tokens, w2v_modelo_cbow)
print(vetor_texto.shape)
print(vetor_texto)

def matriz_vetores(textos, modelo):
    x = len(textos)
    y = 300
    matriz = np.zeros((x,y))

    for i in range(x):
        palavras = tokenizador(textos.iloc[i])
        matriz[i] = combinacao_de_vetores_por_soma(palavras, modelo)

    return matriz

matriz_vetores_treino_cbow = matriz_vetores(artigo_treino.title, w2v_modelo_cbow)
matriz_vetores_teste_cbow = matriz_vetores(artigo_teste.title, w2v_modelo_cbow)
print(matriz_vetores_treino_cbow.shape)
print(matriz_vetores_teste_cbow.shape)