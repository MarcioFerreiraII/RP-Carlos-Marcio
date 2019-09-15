import random as ran
from sklearn import tree
from sklearn.model_selection import train_test_split
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
    
    dici = {} #Aqui eh onde eh criada cada linha da coluna por iteracao
    for k in range(len(xFx[0])): #Percorre a frase...
        palavra = xFx[0][k] #...palavra por palavra
        if len(palavra) < 3: #Ignora palavra com menos de 3 caracteres
            continue
        if palavra not in tabela.columns: #Caso nao exista uma coluna para a palavra...
            tabela[palavra] = [0 for i in range(len(tabela))] #...eh criada uma nova
        dici[palavra] = 1 #E entao seu valor eh 1
    tabela = tabela.append(dici, ignore_index = True) #E finalmente a linha eh adicionada a tabela
    listaDeDados.append(xFx)

arq.close()
tabela = tabela.fillna(0) #Valores nulos sao preenchidos com 0
tabela['classe'] = [linha[1] for linha in listaDeDados] #Coluna com a classe de cada frase eh adicionada a tabela
print(tabela)

#Divide a tabela em treinamento e teste
X = tabela.iloc[:,0:-1]
Y = tabela.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) 

#Treina a arvore
clf_entropy = sk.tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 5)
clf_entropy.fit(X_train, y_train)

predicao = clf_entropy.predict(X_test) #Testa

for i in range(len(predicao)):
    print('\n\nFrase:')
    for j in range(len(X_test.columns)):
        if X_test.iloc[i, j] == 1:
            print(X_test.columns[j], ' ', end = '')
    print('\nPredicao:', predicao[i],'Correto:', y_test.iloc[i])
