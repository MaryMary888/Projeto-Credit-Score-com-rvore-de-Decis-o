# Projeto de Credit Score - Árvore de Decisão
<p>No módulo 17, vocês realizaram a primeira etapa do projeto de crédito de vocês. Então fizeram o tratamendo dos dados, balancearam as classes, transformaram as variáveis categóricas e separam base de treino e teste. Já no módulo 14, aplicaram a base já tratada o algoritmo de Naive Bayes, onde avaliaram os resultados das previsões. Nesse módulo aplicaremos a nossa base o algoritmo da árvore de decisão.</p>

---

#### Sobre separação da base de dados e verificação da compatibilidade entre bases
- As bases de treino e teste estão corretas em número de linhas e colunas.

---

### Documentação do Passo a Passo do Tratamento e Pré-Processamento dos dados do dataframe de Credit-Score
- Para a aplicação do modelo de Árvore de Decisão, inicialmente, no pré-processamento dos dados, realizamos a análise exploratória, a pré-modelagem e modelagem, etapa as quais são feitas a visualização e tratamento dos dados do dataframe, identificamos os dados que estamos lidando e detectando as inconsistências como dados nulos, tipagem dos dados e erros ortográficos, e assim fazer a transformação ou exclusão dos mesmos. No caso deste dataframe, não houve exclusões, e sim, apenas substituições, mudança da tipagem dos dados e tradução do inglês para o português.
- Na nossa próxima etapa, foi realizada a análise univariada onde detectamos a existência de outliers de acordo com os valores mostrados da função describe(), nos trazendo valores como variância, média e mediana. Em seguida, são feitas análises em gráficos para verificar visualmente a presença de outliers e decidir possíveis tratamentos a partir dos insights. Como não haviam outliers no dataframe, não foi necessário nenhum tipo de tratativa.
- Seguindo para a análise bivariada, são analisados gráficos mais complexos trazendo o comportamento e a relação entre 2 variáveis distintas e observando possíveis conexões entre ambas e trazendo suposições do porquê a variável principal possui determinado rótulo/valor. Depois, a matriz de correlação é feita através da função corr() para verificarmos o quão fortemente ligadas as variáveis estão umas com as demais, após a transformação de variáveis categóricas em variáveis numéricas utilizando, por exemplo, LabelEncoder e One Hot para darmos seguimento às análises estatísticas do dataframe.
- Por fim, as bases de treino e de teste são criadas e salvas, após a verificação da igualdade de número de linhas entre as bases e testes e da necessidade de balanceamento dos dados. Nesta situação, como a base apresentava desbalanço, a função SMOTE foi utilizada para o balanceamento por oversampling, a fim de igualar a quantidade de amostras de cada rótulo na base de treino, criando linhas sintéticas.
- Com as bases prontas para uso, o primeiro modelo de machine learning é aplicado para testarmos sua eficácia na base: o naive Bayes. Para análise do modelo, as bases de treino e teste são carregadas e é feita mais uma verificação das bases antes do uso.
- Feito a verificação, a instância criada com GaussianNB() utilizou as bases de treino para praticar e foi avaliada em seguida com as mesmas bases para identificar se o modelo estava prevendo corretamente os rótulos de Crédito e então são calculadas a eficácia e recall do modelo e plotado uma matriz de confusão. O mesmo processo foi feito com as bases de teste.
- A partir dos cálculos e da visualização das matrizes de confusão, entendemos que apesar do modelo ter apresentado bons valores de ajuste e, por mais que o modelo também foi projetado para bases menores, o resultado nos dá fortes indícios que o alto desempenho seja devido ao overfitting, ou seja, o modelo apenas memorizou as linhas apresentadas e não houve aprendizado significativo.
- Na etapa atual, o modelo utilizado é a Árvore de Decisão. Diferente do Naive Bayes, que encara as features da base como independentes, a Árvore de decisão busca fazer a identificação dos rótulos a partir das variáveis que trazem informações mais importantes e fazer um condicionamento lógico com essas features. Na instância escolhemos utilizar os critérios do índice de gini. Em seguida, são calculadas as acurácias na etapa de treino, teste e teste reduzido com apenas as duas features mais importantes da base e também visualizar na base de treino como o modelo realiza a separação dos rótulos plotando uma árvore de decisão.

---

### Aplicação do modelo da Árvore de Decisão nas bases de treino e teste
**Sobre Desempenho de acordo com a acurácia na base de treino**
- A Acurácia do modelo de treino foi equivalente a 100%, mostrando que a generalização dos dados foi perfeita.
 
**Sobre Desempenho de acordo com a acurácia na base de teste**
- O desempenho da base de teste foi equivalente a 0.94, ou seja, o modelo tem taxa de 94% de acerto na classificação da pontuação de Crédito, o que é um desempenho muito bom na generalização dos dados. Porém, é importante ressaltar que, por conta da acurácia da base de treino ter retornado como 1 e a base de teste ter resultado ligeiramente abaixo, esses valores podem indicar overfitting.

**Sobre a visualização da Árvore de Decisão**
- É possível observar claramente a árvore de decisão por possuir pouca profundidade. A árvore possui 5 níveis.

**Sobre as principais features do modelo**
- As features mais importantes utilizadas pelo modelo são as variáveis de Salário Anual e Aquisição de Casa, enquanto as demais tem influência baixa (variável Idade) ou Nula no modelo de decisão.

**Parecer sobre modelo da árvore com apenas as duas features mais relevantes**
- O modelo apenas com as duas features indicadas acima, revela que o modelo teve 92% de acurácia, quase o mesmo valor do teste apresentando todas as features do dataframe. Isso pode nos dizer que as demais features não seriam necessárias para a generalização dos dados, indicando que a base possui mais complexidade que o necessário para a previsão dos níveis de Pontuação de Crédito.

---

## Comparação dos resultados do modelo de Naive Bayes aplicado anteriormente com modelo atual
- Analisando atentamente os modelos acima e realizando a comparação entre ambos os modelos de Naive Bayes e Árvore de Decisão, conseguimos identificar que os modelos apresentam performances quase impecáveis. Porém, não é possível concluir sobre qual modelo é melhor para trabalhar nesta base de dados. O motivo se deve pelos fortes indícios dos dois modelos apresentarem overfitting nas fases de treino e teste, com um desempenho próximo ou igual a 1. O comportamento pertinente de manter uma média bem alta entre os valores de acurácia dso modelos, nos mostra que, na verdade, o modelo apenas memorizou valores e padrões específicos da base de trieno.

- Também é pertinente relembrar que, por mais que os modelos aparentem funcionar bem para bases com pouca quantidade de dados, a falta de diversificação das variáveis para a classificação pode dificultar os treinos e levá-los ao overfitting, tornando os valores de acurácia e outros cálculos de desempenho pouco confiáveis.
