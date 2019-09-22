from sklearn import tree
from sklearn.model_selection import train_test_split
import sklearn as sk
import pandas as pd
from sklearn import naive_bayes as nb
from sklearn.model_selection import KFold
import numpy as np

#Aberttua do arquivo contendo a base de dados
arq = open('novaBase.base', encoding="utf8") #Dados.base é minha base de dados
tabela = pd.DataFrame()
classe = []
for linha in arq:
    linha.strip('\n')
    retiraTabulacao = linha.split('\t')
    classe.append(retiraTabulacao[1])
    retiraTabulacao[0] = retiraTabulacao[0].split() #Transforma a frase em um array de palavras
    dici = {} #Aqui eh onde eh criada cada linha da coluna por iteracao
    for k in range(len(retiraTabulacao[0])): #Percorre a frase...
        palavra = retiraTabulacao[0][k] #...palavra por palavra
        for char in palavra:
            if char in '.,():;?!': #Verifica se ha algum simbolo especial...
                palavra = palavra.replace(char, '') #...e o retira
            elif 65 <= ord(char) <= 90: #Verifica se ha algum caractere maiusculo...
                palavra = palavra.replace(char, chr(ord(char) + 32)) #...e o transforma em minusculo
        if len(palavra) < 3: #Ignora palavra com menos de 3 caracteres
            continue
        if len(palavra) < 3:
            continue
        if palavra not in tabela.columns: #Caso nao exista uma coluna para a palavra...
            tabela[palavra] = [0 for i in range(len(tabela))] #...eh criada uma nova
        dici[palavra] = 1 #E entao seu valor eh 1
    tabela = tabela.append(dici, ignore_index = True) #E finalmente a linha eh adicionada a tabela

arq.close()
tabela = tabela.fillna(0) #Valores nulos sao preenchidos com 0
print(tabela)

#Treina os classificadores
arvoreDecisao = tree.DecisionTreeClassifier(criterion = 'entropy')
bayes = nb.MultinomialNB() #instancia o classificador Naive Bayess MULTINOMIAL

#Validação cruzada, criando
cv = KFold(n_splits=10, shuffle=True)

#Armazenamento dos scores
scoresBayes = []
scoresDT = []

for train_index, test_index in cv.split(tabela, classe):
    X_train, X_test = tabela.iloc[train_index], tabela.iloc[test_index] #Pegando as linhas para teste
    y_train = []
    y_test = []
    #Pegando as labels reais
    for i in range(len(train_index)):
        y_trainTem = classe[train_index[i]]
        y_train.append(y_trainTem)
    for i in range(len(test_index)):
        y_testTem = classe[test_index[i]]
        y_test.append(y_testTem)
    bayes.fit(X_train, y_train) #Treinando a arvore
    arvoreDecisao.fit(X_train, y_train) #Treinando o Bayes
    # Guardando os scores
    scoresBayes.append(bayes.score(X_test, y_test))
    scoresDT.append(arvoreDecisao.score(X_test, y_test))

print("Precisões do Naive Bayes:", scoresBayes)
print("Precisões da Arvore de Decisoes:", scoresDT)

#Imprime a media, o desvio padrao e a accuracia do teste de validaçao cruzada
print("\nMedia Arvore de Decisao (Validacao Cruzada):", np.mean(scoresDT))
print("Desvio padrão Arvore de Decisao (Validacao Cruzada):", np.std(scoresDT))

print("\nMedia Naive Bayes (Validacao Cruzada):", np.mean(scoresBayes))
print("Desvio padrão Naive Bayes (Validacao Cruzada):", np.std(scoresBayes))

#Testa uma frase
print("\nDigite uma frase: ")
frase = input().split()
novaFrase = pd.DataFrame(columns = tabela.columns)
dici = {}
for k in range(len(frase)):
    palavra = frase[k]
    for char in palavra:
        if char in '.,():;?!':  # Verifica se ha algum simbolo especial...
            palavra = palavra.replace(char, '')  # ...e o retira
        elif 65 <= ord(char) <= 90:  # Verifica se ha algum caractere maiusculo...
            palavra = palavra.replace(char, chr(ord(char) + 32))  # ...e o transforma em minusculo
    if palavra in tabela.columns:
        dici[palavra] = 1
novaFrase = novaFrase.append(dici, ignore_index = True)
novaFrase = novaFrase.fillna(0)  #Valores nulos sao preenchidos com 0
predictBayes = bayes.predict(novaFrase)
predictDT = arvoreDecisao.predict(novaFrase)

if (np.mean(scoresBayes) > np.mean(scoresDT)):
    print(predictBayes)
else:
    print(predictDT)

