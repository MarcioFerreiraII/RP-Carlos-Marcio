import random as ran
import copy
import sklearn as sk

#Aberttua do arquivo contendo a base de dados
arq = open('dados.base', 'r') #Dados.base é minha base de dados
listaDeDados = []
i = 0
for linha in arq:
    retiraTabulacao = linha.split('\t')
    xFx = (retiraTabulacao[0], retiraTabulacao[1])
    listaDeDados.append(xFx)
    pos.append

arq.close()
copiaListaOriginal = copy.copy(listaDeDados)
ran.shuffle(listaDeDados)

sk.tree.DecisionTreeClassifier()

for dado in listaDeDados:
    print(dado[1])
