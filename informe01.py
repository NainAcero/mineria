# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:35:50 2021

@author: NAIN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error

#IMPORTAMOS EL DATASET
#-----------------------------------------------------------------------------------------------------------------
df_customer = pd.read_csv('dataset/Ecommerce Customers.csv')
df_customer.drop(columns=['Email', 'Address', 'Avatar'], inplace=True)

#DIVIDIMOS EN TRAIN Y TEST
#-----------------------------------------------------------------------------------------------------------------
train = df_customer.drop(columns=['Yearly Amount Spent'])
test = df_customer['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2)

#REALIZAMOS EL ALGORITMO Y COMPROBAMOS RESULTADOS
#-----------------------------------------------------------------------------------------------------------------

#Instanciamos el algoritmo
svr_2 = SVR(kernel='linear', C=10, cache_size=800, )

#Entrenamos
svr_2.fit(X_train, y_train)

#Predecimos
train_preds = svr_2.predict(X_train)
test_preds = svr_2.predict(X_test)

#Vemos el error del algoritmo a la hora de predecir, para comprobar si a ajustado bien.
print('MAE in train:', mean_absolute_error(train_preds, y_train))
print('RMSE in train:', np.sqrt(mean_squared_error(train_preds, y_train)))
print('MAE in test:', mean_absolute_error(test_preds, y_test))
print('RMSE in test:', np.sqrt(mean_squared_error(test_preds, y_test)))

# VEMOS LOS COEFICIENTES DEL ALGORITMO
#-----------------------------------------------------------------------------------------------------------------
pd.DataFrame(svr_2.coef_, columns=X_train.columns).T
print(pd.DataFrame(svr_2.coef_, columns=X_train.columns).T)

print()
print('DATOS DEL MODELO VECTORES DE SOPORTE REGRESIÓN')
print()
print('Precisión del modelo:')
print(svr_2.score(X_train, y_train))
