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