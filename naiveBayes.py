import random as ran
from sklearn import tree
from sklearn.model_selection import train_test_split
import sklearn as sk
from sklearn import naive_bayes as nb
import pandas as pd


# Aberttua do arquivo contendo a base de dados
arq = open('dados.base', 'r')  # Dados.base é minha base de dados
listaDeDados = []
listaDeDadosCopia = []
tabela = pd.DataFrame()
for linha in arq:
    retiraTabulacao = linha.split('\t')
    listaDeDadosCopia.append((retiraTabulacao[0], retiraTabulacao[1]))

for linha in listaDeDadosCopia:
    for char in linha[0]:  # Percorre a frase da linha especifica
        if 65 <= ord(char) <= 90:  # Verifica se ha algum caractere maiusculo...
            linhaTemp = linha[0].replace(char, chr(ord(char) + 32))  # ...e o transforma em minusculo
        elif char in '.,():;?!':  # Verifica se ha algum simbolo especial...
            linhaTemp = linha[0].replace(char, '')  # ...e o retira
    linhaTemp = linhaTemp.split()  # Transforma a frase em um array de palavras
    xFx = [linhaTemp, linha[1]]

    dici = {}  # Aqui eh onde eh criada cada linha da coluna por iteracao
    for k in range(len(xFx[0])):  # Percorre a frase...
        palavra = xFx[0][k]  # ...palavra por palavra
        if len(palavra) < 3:  # Ignora palavra com menos de 3 caracteres
            continue
        if palavra not in tabela.columns:  # Caso nao exista uma coluna para a palavra...
            tabela[palavra] = [0 for i in range(len(tabela))]  # ...eh criada uma nova
        dici[palavra] = 1  # E entao seu valor eh 1
    tabela = tabela.append(dici, ignore_index=True)  # E finalmente a linha eh adicionada a tabela
    listaDeDados.append(xFx)
arq.close()

tabela = tabela.fillna(0)  # Valores nulos sao preenchidos com 0
tabela['classe'] = [linha[1] for linha in listaDeDados]  # Coluna com a classe de cada frase eh adicionada a tabela
print(tabela)

ran.shuffle(listaDeDadosCopia)

treinamento = listaDeDadosCopia[0:250]
teste = listaDeDadosCopia[251:450]

#Divide a tabela em treinamento e teste
X = tabela.iloc[:,0:-1]
Y = tabela.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#Treina o classificador
bayes = nb.MultinomialNB() #instancia o classificador Gaussiano
bayes.fit(X_train, y_train)

posteriori = bayes.predict(X_test)
probabilidades = bayes.predict_proba(X_test)

for i in range(len(posteriori)):
    print('\nFrase:')
    for j in range(len(X_test.columns)):
        if X_test.iloc[i, j] == 1:
            print(X_test.columns[j], ' ', end = '')
    print('\nPredicao:', posteriori[i],'Correto:', y_test.iloc[i], 'Posteriori:', probabilidades[i])




