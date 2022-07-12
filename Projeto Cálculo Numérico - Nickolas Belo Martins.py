#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
random.seed(42)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#lê os dados
data = pd.read_csv("BostonHousing.csv")

#Função para a regressão linear múltipla e o coeficiente R2 
def regressao_coeficienteR2():
    
    # valor a ser predito
    ylabel = data.columns[-1]
    print("Número de linhas e colunas:", data.shape, "\n")
    data.head(10)

    dados = data.to_numpy()
    nrow,ncol = dados.shape
    y = dados[:,-1]
    X = dados[:,0:ncol-1]


    # divide o conjunto em treinamento e teste
    p = 0.3 # fracao e elementos no conjnto de teste
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = p, random_state = 42)

    # modelo de regressão linear múltipla
    print("Gráfico Regressão Linear Múltipla")
    lm = LinearRegression()
    lm.fit(x_train, y_train)

    y_pred = lm.predict(x_test)

    fig = plt.figure()
    l = plt.plot(y_pred, y_test, 'bo')
    plt.setp(l, markersize=10)
    plt.setp(l, markerfacecolor='C0')
    plt.ylabel("y", fontsize=15)
    plt.xlabel("Predição", fontsize=15)
    # mostra os valores preditos e originais
    xl = np.arange(min(y_test), 1.2*max(y_test),(max(y_test)-min(y_test))/10)
    yl = xl
    plt.plot(xl, yl, 'r--')
    plt.show(True)

    #Coeficiente de determinação R2 
    R2 = r2_score(y_test, y_pred)
    print('Coeficiente R2 da regressão linear múltipla :', R2, "\n\n")

def Pearson():
    corr = data.corr()
    #Plotando matriz de correlação de Pearson usando Matplotlib
    plt.figure(figsize=(12,10))
    sns.heatmap(corr,annot=True) # matriz de correlação
    

def Spearman():
    #Plotando matriz de correlação de Spearman usando Matplotlib
    correlacao= data.corr(method='spearman')
    plt.figure(figsize=(12,10))
    sns.heatmap(correlacao,annot=True) # matriz de correlação Spearman
   
def main():
    
    regressao_coeficienteR2()
    print("Matriz de correlação de Pearson e matriz de correlação de Spearman respectivamente\n")
    Pearson()
    Spearman()
    
main()    

