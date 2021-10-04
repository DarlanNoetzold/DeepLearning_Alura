import pandas as pd
import spacy

dados_treino = pd.read_csv("/content/drive/My Drive/curso_word2vec/treino.csv")
dados_treino.sample(5)

nlp = spacy.load("pt_core_news_sm")

texto = "Rio de Janeiro é uma cidade maravilhosa"
doc = nlp(texto)
doc
