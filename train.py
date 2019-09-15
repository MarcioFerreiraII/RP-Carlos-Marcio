import random as ran
import copy
from sklearn import tree
import sklearn as sk
import pandas as pd

#Aberttua do arquivo contendo a base de dados
arq = open('dados.base', 'r') #Dados.base é minha base de dados
listaDeDados = []
tabela = pd.DataFrame()
for linha in arq:
    retiraTabulacao = linha.split('\t')
    for char in retiraTabulacao[0]: #Percorre a frase da linha especifica
        if char in '.,():;?!': #Verifica se ha algum simbolo especial...
            retiraTabulacao[0] = retiraTabulacao[0].replace(char, '') #...e o retira
    retiraTabulacao[0] = retiraTabulacao[0].split() #Transforma a frase em um array de palavras
    xFx = [retiraTabulacao[0], retiraTabulacao[1]]
    
    dici = {}
    for k in range(len(xFx[0])):
        palavra = xFx[0][k]
        if len(palavra) < 3:
            continue
        if palavra not in tabela.columns:
            tabela[palavra] = [0 for i in range(len(tabela))]
        dici[palavra] = 1
    tabela = tabela.append(dici, ignore_index = True)
    listaDeDados.append(xFx)

arq.close()
tabela = tabela.fillna(0)
tabela['classe'] = [linha[1] for linha in listaDeDados]
print(tabela)
sk.tree.DecisionTreeClassifier()
