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