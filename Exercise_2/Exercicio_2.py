#!/usr/bin/env python
# coding: utf-8

# In[141]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from numpy.random import uniform
from numpy.linalg import norm
from scipy.stats import mode


# # Exercício 1

# Este exercício se refere à classificação do gênero do locutor a partir de trechos de voz.

# In[142]:


dados_voz = pd.read_csv('dados_voz_genero.csv')


# Obtendo informações com relação aos dados

# In[143]:


#Knowing the data
dados_voz.head()


# In[144]:


dados_voz.info()


# ## Analisando os Dados

# Neste ponto temos que nenhum dos 19 atributos possui dados ausentes. É importante citar que a última coluna "label", o valor 1 corresponde ao gênero masculino e 0 ao feminino.

# In[145]:


#Remove the unnamed collumn 
dados_voz = dados_voz.drop(['Unnamed: 0'], axis=1)
dados_voz.corr().style.background_gradient(cmap='coolwarm').set_precision(2)


# In[146]:


dados_voz.hist(bins=50, figsize=(20, 15))


# Aqui vamos separar 20% dos dados como conjunto de teste

# In[147]:


#Creating the test and training data
#X_train, X_test, Y_train, Y_test = train_test_split(dados_voz.loc[:, 'sd':'modindx'], dados_voz.loc[:, 'label'], 
#                                                    test_size=0.2, random_state=10)
X_train, X_test, Y_train, Y_test = dados_voz.loc[0:2533, 'sd':'modindx'], dados_voz.loc[2534:, 'sd':'modindx'], dados_voz.loc[0:2533, 'label'], dados_voz.loc[2534:, 'label']
#
#Y_train.shape


# In[148]:


def sigmoid(phi, W):
    z = np.matmul(phi, W)

    return 1/(1 + np.exp(-z))

#This function returns the value and the gradient of the logistic regression cost function 
#for some given parameters
def logistic_regression_cost_function(phi, y, w):
    y_est = sigmoid(phi, w)
    
    J = -np.mean(y*np.log(y_est) + (1-y)*np.log(1-y_est))
    J_grad = -(np.matmul((y-y_est).T, phi))/y.shape[0]

    return J, J_grad

def gradient_descent(phi, y, lr, iter_max, tol):
    iterations = 0
    del_J = 1e9
    k = phi.shape[1]
    w = np.zeros(k)
    J_old = 0
    J_list = []
    
    while del_J > tol and iterations < iter_max:
        J_new, J_grad = logistic_regression_cost_function(phi, y, w)
        J_list.append(J_new)
        w = w - lr*J_grad

        del_J = np.absolute(J_new - J_old)
        J_old = J_new
        
        iterations += 1
        print("\nIteration: " + str(iterations) + "\nCost Function: " + str(J_new))
        
    print('Optimization is Over!')
    print('Number of Iterations: ', iterations)
    print('Cost Function Variation: ', del_J)
    
    return w, J_list


# É possível ver por meio dos histogramas que os atributos possuem diferentes faixas de valores. Assim, para facilitar o processo de otimização, faz-se a normalização dos dados.

# In[149]:


#Adding ones to allow the w0 optimization
X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

X_train_norm = np.copy(X_train)
X_test_norm = np.copy(X_test)

X_train_norm[:, 1:] = normalize(X_train[:, 1:].T).T
X_test_norm[:, 1:] = normalize(X_test[:, 1:].T).T

Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()


# Agora vamos realizar o treinamento para a obtenção dos vetores W.

# In[150]:


#Training the model
w, J_iteration = gradient_descent(X_train_norm, Y_train, 4, 50000, 1e-6)

#Obtaining the predicted values
y_est_prob_val = sigmoid(X_test_norm, w)

f1_validation = []
#Obtaining the F1-Curve
threshold_values = np.arange(0, 1.001, 0.01)
for threshold in threshold_values:
    y_est_val = y_est_prob_val > threshold
    y_est_val = y_est_val.astype(int) #Convert the boolean to integer

    f1_validation.append(f1_score(Y_test, y_est_val))  


# In[151]:


plt.plot(threshold_values, f1_validation)
plt.xlabel('Threshold')
plt.ylabel('F1 Metric')
plt.title('F1 Curve')
threshold_max = f1_validation.index(np.max(f1_validation))/100
print("Threshold que maximiza F1: " + str(threshold_max))
print("F1 Máximo: " + str(np.max(f1_validation)))


# In[152]:


plt.plot(J_iteration)
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.title('Cost Function vs Iterations')


# In[153]:


fp, tp, _ = roc_curve(Y_test, y_est_prob_val)
area = auc(fp, tp)


# In[154]:


plt.plot(fp, tp, color='darkorange', label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.legend(loc="lower right")
plt.title('ROC Curve')


# Calculando a Matriz de confusão para o menor threshold que foi de 0.11

# In[155]:


y_est_val = y_est_prob_val > threshold_max
y_est_val = y_est_val.astype(int) #Convert the boolean to integer

cm = confusion_matrix(Y_test, y_est_val)

tn, fp, fn, tp = cm.ravel()

print(cm)
accuracy = accuracy_score(Y_test, y_est_val)
print("Accuracy: " + str(accuracy))


# # Exercício 2

# Este exercício realiza a classificação multi-classe utilizando os algoritmos de regressão logística e k-Nearest Neighbours. 
# A base de dados utilizada corresponde aos atributos nos domínios do tempo e da frequência extraídos de sinais de acelerômetro e giroscópio de um smartphone.
# 
# Os rótulos correspondem a: 
# 0 - Caminhada
# 1 - Subindo escadas
# 2 - Descendo escadas
# 3 - Sentado
# 4 - Em pé
# 5 - Deitado

# In[156]:


X_train = pd.read_fwf('/home/felippe/Área de Trabalho/Felippe/Mestrado/Machine_Learning/IA006/Exercicio_2/har_smartphone/X_train.txt').to_numpy()
X_test = pd.read_fwf('/home/felippe/Área de Trabalho/Felippe/Mestrado/Machine_Learning/IA006/Exercicio_2/har_smartphone/X_test.txt').to_numpy()

Y_train = pd.read_fwf('/home/felippe/Área de Trabalho/Felippe/Mestrado/Machine_Learning/IA006/Exercicio_2/har_smartphone/y_train.txt').to_numpy()
Y_test = pd.read_fwf('/home/felippe/Área de Trabalho/Felippe/Mestrado/Machine_Learning/IA006/Exercicio_2/har_smartphone/y_test.txt').to_numpy()


# Agora vamos criar algumas funções para a implementação do algoritmo de regressão logística utilizando softmax.

# In[157]:


def softmax_estimation(X, w):
    z = np.exp(np.matmul(X, w))
    z_sum = np.expand_dims(np.sum(z, axis=1), axis=0)
    y_est = z/z_sum.T
    
    return y_est

def convert_y_softmax(y):
    y_labels = np.unique(y)
    N = y.shape[0]
    k = y_labels.shape[0]
    y_softmax = np.zeros((N, k))
    
    
    for i in range(N):
        y_softmax[i, y[i]-1] = 1
    
    return y_softmax

def logistic_regression_multi_class_cost_function(X, w, y):
    y_est = softmax_estimation(X, w)
    y_softmax = convert_y_softmax(y)

    J = -np.mean(np.sum( y_softmax*np.log(y_est), axis=1 )) 
    J_grad = -(1/X.shape[0])*np.matmul(X.T, (y_softmax - y_est))
    
    return J, J_grad

def gradient_descent_softmax(X, y, lr, iter_max, tol):
    iterations = 0
    del_J = 1e9
    k = X.shape[1]
    w = np.zeros([k, np.unique(y).shape[0]])
    J_old = 0
    J_list = []
    
    while del_J > tol and iterations < iter_max:
        J_new, J_grad = logistic_regression_multi_class_cost_function(X, w, y)
        J_list.append(J_new)
        w = w - lr*J_grad

        del_J = np.absolute(J_new - J_old)
        J_old = J_new
        
        iterations += 1
        
        print("\nIteration: " + str(iterations) + "\nCost Function: " + str(J_new))
        
    print('Optimization is Over!')
    print('Number of Iterations: ', iterations)
    print('Cost Function Variation: ', del_J)
    
    return w, J_list


# In[158]:


#Adding ones to allow the w0 optimization
X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

X_train_norm = X_train
X_test_norm = X_test

X_train_norm[:, 1:] = normalize(X_train[:, 1:].T).T
X_test_norm[:, 1:] = normalize(X_test[:, 1:].T).T


# In[159]:


#Training the model
w, J_iteration = gradient_descent_softmax(X_train_norm, Y_train, 11, 12000, 1e-6)


# In[ ]:


plt.plot(J_iteration)
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.title('Cost Function vs Iterations')


# Agora vamos avaliar os resultados para o conjunto de teste

# In[ ]:


y_test_estimation = softmax_estimation(X_test, w)
y_test_prediction = np.argmax(y_test_estimation, axis=1)+1

conf_matrix = confusion_matrix(Y_test, y_test_prediction)
F1_score = f1_score(Y_test, y_test_prediction, average='micro')
#fp, tp, _ = roc_curve(Y_test, y_test_prediction)
print(F1_score)
print(conf_matrix)


# Para construir o algoritmo de K-vizinhos mais próximos se faz necessário calcular a distância entre o novo padrão de entrada e todas os outros padrões de treinamento. Com isso pode-se criar um dicionário ou um array bidimensional onde as linhas são os padrões de enrada e as k primeiras colunas são os atributos e a última coluna corresponde a saída. Assim ordena-se o vetor com relação aos seu valor de distância e pega-se os padrões que contém as k menores distâncias.
# 
# Neste ponto pode-se utilizar duas abordagens: (i) o voto majoritário, ou seja, calcular a quantidade de labels de cada label, entre o conjunto de k vizinhos mais próximos; (ii) Média ponderada pela distância: calcula-se a média dos rótulos em questão, ponderados pelo inverso da distância entre os rótulos. PORÉM NESSE PONTO TEMOS UM PROBLEMA: O RESULTADO NÃO SERÁ MAIS UM INTEIRO, ENTÃO COMO CALCULAR O VALOR DO RÓTULO??

# In[ ]:


def KNN(X, y, new_x, k):
    dist1 = X - new_x
    dist_matrix = np.expand_dims(norm(dist1, axis=1), axis=0).T
    X_and_distance = np.concatenate((X, y), axis=1)
    X_and_distance = np.concatenate((X_and_distance, dist_matrix), axis = 1)
    X_and_distance_ordered = X_and_distance[X_and_distance[:, X_and_distance.shape[1]-1].argsort()]
    
    X_KNN = X_and_distance_ordered[0:k ,:]
    
    y_KNN = X_KNN[:, X_KNN.shape[1]-2]
    
    return mode(y_KNN)[0]


# In[ ]:


X_train = X_train[:, 1:]
X_test = X_test[:, 1:]
K_values = [1, 3, 6, 10, 30, 60, 100]

Y_predicted = np.zeros((X_test.shape[0], len(K_values)))

iterations = 0
for k in K_values:
    print("K Value: " + str(k))
    for i in range(X_test.shape[0]):
        print("Test input: " + str(i))
        Y_predicted[i, iterations] = KNN(X_train, Y_train, X_test[i, :], k)
    iterations += 1


# In[ ]:


conf_matrix_1 = confusion_matrix(Y_test, Y_predicted[:, 0])
F1_score_1 = f1_score(Y_test, Y_predicted[:, 0], average='micro')

print(conf_matrix_1)
print(F1_score_1)

conf_matrix_3 = confusion_matrix(Y_test, Y_predicted[:, 1])
F1_score_3 = f1_score(Y_test, Y_predicted[:, 1], average='micro')

print(conf_matrix_3)
print(F1_score_3)

conf_matrix_6 = confusion_matrix(Y_test, Y_predicted[:, 2])
F1_score_6 = f1_score(Y_test, Y_predicted[:, 2], average='micro')

print(conf_matrix_6)
print(F1_score_6)

conf_matrix_10 = confusion_matrix(Y_test, Y_predicted[:, 3])
F1_score_10 = f1_score(Y_test, Y_predicted[:, 3], average='micro')

print(conf_matrix_10)
print(F1_score_10)

