from sklearn import tree
from sklearn.model_selection import train_test_split
import sklearn as sk
import pandas as pd
from sklearn import naive_bayes as nb
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#Aberttua do arquivo contendo a base de dados
arq = open('novaBase.base', encoding="utf8") #Dados.base é minha base de dados
tabela = pd.DataFrame()
classe = []
for linha in arq:
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
        if palavra not in tabela.columns: #Caso nao exista uma coluna para a palavra...
            tabela[palavra] = [0 for i in range(len(tabela))] #...eh criada uma nova
        dici[palavra] = 1 #E entao seu valor eh 1
    tabela = tabela.append(dici, ignore_index = True) #E finalmente a linha eh adicionada a tabela

arq.close()
tabela = tabela.fillna(0) #Valores nulos sao preenchidos com 0
print(tabela)

#Divide a tabela em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(tabela, classe, test_size=0.2)

#Treina os classificadores
arvoreDecisao = tree.DecisionTreeClassifier(criterion = 'entropy')
arvoreDecisao.fit(X_train, y_train)
bayes = nb.MultinomialNB() #instancia o classificador Naive Bayess MULTINOMIAL
bayes.fit(X_train, y_train)

predicaoArvoreDecisao = arvoreDecisao.predict(X_test) #Testa

predicaoBayes = bayes.predict(X_test)
probabilidadesBayes = bayes.predict_proba(X_test)

#imprime o resultado do teste
for i in range(len(predicaoArvoreDecisao)):
    print('\nFrase:')
    for j in range(len(X_test.columns)):
        if X_test.iloc[i, j] == 1:
            print(X_test.columns[j], ' ')
    print('\nPredicao Arvore de decisao:', predicaoArvoreDecisao[i])
    print('Predicao NaiveBayes:', predicaoBayes[i])
    print('Correto:', y_test[i])

print('\nPrecisao Arvore de decisao:', str(sk.metrics.accuracy_score(y_test, predicaoArvoreDecisao) * 100) + '%')
print('Precisao Naive Bayes:', str(sk.metrics.accuracy_score(y_test, predicaoBayes) * 100) + '%')

#Cria arquivo em pdf da arvore de decisao
dot_data = tree.export_graphviz(arvoreDecisao, out_file = None, feature_names = tabela.columns, class_names = arvoreDecisao.classes_, filled = True, rounded = True, special_characters = True)
graph = graphviz.Source(dot_data)
graph.render('grafico')
