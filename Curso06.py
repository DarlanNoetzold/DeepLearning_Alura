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