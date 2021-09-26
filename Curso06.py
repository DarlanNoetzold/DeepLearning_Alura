import pandas as pd
dados_portugues = pd.read_csv("stackoverflow_portugues.csv")
dados_portugues.head()

questao_portugues = dados_portugues.Questão[5]
print(questao_portugues)

dados_ingles = pd.read_csv("stackoverflow_ingles.csv")
dados_ingles.head()

questao_ingles = dados_ingles.Questão[0]
print(questao_ingles)

import re
re.findall(r"<.*?>",questao_portugues)
print(questao_portugues)

texto_teste = re.sub(r"<.*?>","  T----E----S----T----E  ",questao_portugues)
print(texto_teste)

re.search(r"70","18728736187263817628631872638716283670")
regex = re.compile(r"70")
regex.search("18728736187263817628631872638716283670")
from timeit import timeit
setup = """import re"""
timeit("""re.search(r"70","18728736187263817628631872638716283670")""", setup)
setup = """import re
regex = re.compile(r"70")"""

timeit("""regex.search("18728736187263817628631872638716283670")""", setup)

def remover(textos, regex):
    if type(textos) == str:
        return regex.sub("", textos)
    else:
        return [regex.sub("", texto) for texto in textos]


regex_html = re.compile(r"<.*?>")
questao_sem_tag = remover(questao_ingles, regex_html)
print(questao_sem_tag)

print(questao_ingles)
def substituir_codigo(textos, regex):
    if type(textos) == str:
        return regex.sub("CODE", textos)
    else:
        return [regex.sub("CODE", texto) for texto in textos]

regex_codigo = re.compile(r"<code>(.|(\n))*?</code>")
questoes_port_sem_code = substituir_codigo(dados_portugues.Questão,
                                           regex_codigo)
questoes_port_sem_code_tag = remover(questoes_port_sem_code, regex_html)


dados_portugues["sem_code_tag"] = questoes_port_sem_code_tag

questoes_ing_sem_code = substituir_codigo(dados_ingles.Questão,
                                           regex_codigo)
questoes_ing_sem_code_tag = remover(questoes_ing_sem_code, regex_html)


dados_ingles["sem_code_tag"] = questoes_ing_sem_code_tag
dados_ingles.head()

regex_pontuacao = re.compile(r"[^\w\s]")
def minusculo(textos):
    if type(textos) == str:
        return textos.lower()
    else:
        return [texto.lower() for texto in textos]

regex_digitos = re.compile(r"\d+")
print(remover("Alura \n 1234 Caelum 1234", regex_digitos))

regex_espaco = re.compile(r" +")
regex_quebra_linha = re.compile(r"(\n)")


def substituir_por_espaco(textos, regex):
    if type(textos) == str:
        return regex.sub(" ", textos)
    else:
        return [regex.sub(" ", texto) for texto in textos]


print(substituir_por_espaco("Alura \n \n     Caleum", regex_quebra_linha))

questoes_port_sem_pont = remover(dados_portugues.sem_code_tag,
                                 regex_pontuacao)
questoes_port_sem_pont_minus = minusculo(questoes_port_sem_pont)
questoes_port_sem_pont_minus_dig = remover(questoes_port_sem_pont_minus,
                                          regex_digitos)


questoes_port_sem_quebra_linha = substituir_por_espaco(questoes_port_sem_pont_minus_dig,
                                                       regex_quebra_linha)
questoes_port_sem_espaco_duplicado = substituir_por_espaco(questoes_port_sem_quebra_linha,
                                                          regex_espaco)

dados_portugues["questoes_tratadas"] = questoes_port_sem_espaco_duplicado

questoes_ing_sem_pont = remover(dados_ingles.sem_code_tag,
                                 regex_pontuacao)
questoes_ing_sem_pont_minus = minusculo(questoes_ing_sem_pont)
questoes_ing_sem_pont_minus_dig = remover(questoes_ing_sem_pont_minus,
                                          regex_digitos)


questoes_ing_sem_quebra_linha = substituir_por_espaco(questoes_ing_sem_pont_minus_dig,
                                                       regex_quebra_linha)
questoes_ing_sem_espaco_duplicado = substituir_por_espaco(questoes_ing_sem_quebra_linha,
                                                          regex_espaco)

dados_ingles["questoes_tratadas"] = questoes_ing_sem_espaco_duplicado

from nltk.util import bigrams
texto_teste = "alura"
print(list(bigrams(texto_teste)))

from nltk.lm.preprocessing import pad_both_ends

print(list(bigrams(pad_both_ends(texto_teste, n = 2))))

from sklearn.model_selection import train_test_split

port_treino, port_teste = train_test_split(dados_portugues.questoes_tratadas,
                                          test_size = 0.2,
                                          random_state = 123)

ing_treino, ing_teste = train_test_split(dados_ingles.questoes_tratadas,
                                          test_size = 0.2,
                                          random_state = 123)
todas_questoes_port = ' '.join(port_treino)
from nltk.tokenize import WhitespaceTokenizer

todas_palavras_port = WhitespaceTokenizer().tokenize(todas_questoes_port)
print(todas_palavras_port)

from nltk.lm.preprocessing import padded_everygram_pipeline


port_treino_bigram, vocab_port = padded_everygram_pipeline(2,
                                                           todas_palavras_port)

from nltk.lm.preprocessing import padded_everygram_pipeline


port_treino_bigram, vocab_port = padded_everygram_pipeline(2,
                                                           todas_palavras_port)

from nltk.lm import MLE

modelo_port = MLE(2)
modelo_port.fit(port_treino_bigram, vocab_port)

modelo_port.generate(num_words=6)
from nltk.lm import NgramCounter

modelo_port.counts[['m']].items()

texto = "good morning"
palavras = WhitespaceTokenizer().tokenize(texto)
palavras_fakechar = [list(pad_both_ends(palavra, n = 2)) for palavra in palavras]
palavras_bigramns = [list(bigrams(palavra)) for palavra in palavras_fakechar]
print(palavras_bigramns)

print(palavras_bigramns[0])
print(modelo_port.perplexity(palavras_bigramns[0]))
print(modelo_port.perplexity(palavras_bigramns[1]))


def treinar_modelo_mle(lista_textos):
    todas_questoes = ' '.join(lista_textos)
    todas_palavras = WhitespaceTokenizer().tokenize(todas_questoes)
    bigrams, vocabulario = padded_everygram_pipeline(2, todas_palavras)
    modelo = MLE(2)
    modelo.fit(bigrams, vocabulario)

    return modelo

modelo_port_2 = treinar_modelo_mle(port_treino)
print(modelo_port_2.perplexity(palavras_bigramns[0]))
print(modelo_port_2.perplexity(palavras_bigramns[1]))
modelo_ing = treinar_modelo_mle(ing_treino)

print(modelo_ing.perplexity(palavras_bigramns[0]))
print(modelo_ing.perplexity(palavras_bigramns[1]))


def calcular_perplexidade(modelo, texto):
    perplexidade = 0
    palavras = WhitespaceTokenizer().tokenize(texto)
    palavras_fakechar = [list(pad_both_ends(palavra, n=2)) for palavra in palavras]
    palavras_bigramns = [list(bigrams(palavra)) for palavra in palavras_fakechar]

    for palavra in palavras_bigramns:
        perplexidade += modelo.perplexity(palavra)

    return perplexidade

print(calcular_perplexidade(modelo_ing, "good morning"))
print(calcular_perplexidade(modelo_port, port_teste.iloc[0]))

port_teste.iloc[0]
print(calcular_perplexidade(modelo_ing, port_teste.iloc[0]))
