import pandas as pd
dados_portugues = pd.read_csv("stackoverflow_portugues.csv")
dados_portugues.head()

questao_portugues = dados_portugues.Questão[5]
print(questao_portugues)