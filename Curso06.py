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