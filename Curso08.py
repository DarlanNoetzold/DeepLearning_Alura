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