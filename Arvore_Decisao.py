import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Arquivos de treino e teste carregados
x_teste = pd.read_csv('x_teste.csv', delimiter=',')
y_teste = pd.read_csv('y_teste.csv', delimiter=',')
x_treino = pd.read_csv('x_treino_balanced.csv', delimiter=',')
y_treino = pd.read_csv('y_treino_balanced.csv', delimiter=',')
games = pd.read_csv('games_dashboard.csv', delimiter=',')
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(games.head(30).to_string())
print(games.dtypes)

# Verificando disposição das bases
treino_balanceado = y_treino.value_counts()
print('Balanceamento do y_treino: \n', treino_balanceado, '\n')
teste_balanceado = y_teste.value_counts()
print('Balanceamento do y_teste: \n', teste_balanceado)

print(x_treino, '\n\n Tamanho de x_treino: ', x_treino.shape)
print(x_teste, '\n\n Tamanho de x_teste: ', x_teste.shape)
print(y_treino, '\n\n Tamanho de y_treino: ', y_treino.shape)
print(y_teste, '\n\n Tamanho de y_teste: ', y_teste.shape)

# Instanciando Árvore de Decisão
arvore_credito = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
arvore_credito.fit(x_treino, y_treino)
print(arvore_credito)
# para verificar parâmetros de qualquer função, use print(NOME_FUNÇÃO.get_params())

# Verificando acurácia do modelo de treino
prev = arvore_credito.predict(x_treino)
acuracia = accuracy_score(y_treino, prev)
print('\nAcurácia do modelo de treino: ', acuracia)

# Verificando acurácia do modelo de teste
prevt = arvore_credito.predict(x_teste)
acuraciat = accuracy_score(y_teste, prevt)
print('\nAcurácia do modelo de teste: ', acuraciat)

# Plotando Árvore de Decisão
plt.figure(figsize=(6, 7))
plot_tree(arvore_credito, filled = True, feature_names = x_treino.columns, class_names = ['0','1','2'])
plt.show()

# identificando features importantes pelo gráfico
imp_features = arvore_credito.feature_importances_
nome_feature = x_treino.columns

plt.figure(figsize = (8, 6))
plt.barh(nome_feature, imp_features)
plt.title('Importância das features na Árvore de Decisão')
plt.xlabel('Nível de Importância')
plt.ylabel('Features')
plt.show()

# Modelo Árvore de Decisão com apenas 2 features
x_treino_reduz = x_treino[['Aquisicao_Casa','Salario_Anual']]
x_teste_reduz = x_teste[['Aquisicao_Casa','Salario_Anual']]

arvore_credito_reduz = DecisionTreeClassifier(criterion = 'gini',  random_state = 0)
arvore_credito_reduz.fit(x_treino_reduz, y_treino)
prev_reduz = arvore_credito_reduz.predict(x_teste_reduz)

# Verificando acurácia do modelo de teste com 2 features
prevt_reduz = arvore_credito_reduz.predict(x_teste_reduz)
acuraciat_reduz = accuracy_score(y_teste, prevt_reduz)
print('\nAcurácia do modelo de teste reduzido: ', acuraciat_reduz)


