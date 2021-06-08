# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics 
import pickle

#loading csv data
gold_data=pd.read_csv('/Users/sristichowdhury/Desktop/gld_price_data.csv')
#print(gold_data.head(5))


#Column data detail
# date , spx=snp_index(capitalization index of the companies-stock valye)
# gld = present gold prices on that current date =y =which will be predicting depending on stock prices
# uso = united states oil price for this dates
# slv =silver price
# eur/usd = euro/dollar
# 1 euro = 1.47 dollar

correlation = gold_data.corr()
#gives the correlation btwn cols
plt.figure(figsize=(8,8))
#cbar=colour bar(rightmostbar) ,cmap=Colour map square , fmt=.1f ( one flaoting points),
#annotations = name of cols ,cmap=color map=blue,
sns.heatmap(correlation,cbar=True,square=True,fmt=".1f",annot=True,annot_kws={'size':8},cmap='Blues')


sns.distplot(gold_data['GLD'],color='red')
#majroity is in 120


X=gold_data.drop(['Date','GLD'],axis=1) #axis=1,dropping cols
Y=gold_data['GLD']

#print(X)
#print(Y)


#Separating into testing data and training data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2) 
#test_size=20%/10%    here,we have 20% as the test data size
#give any value in random_State

#Random forest - ensemble model - consits of many Decision trees joined together 
regressor=RandomForestRegressor(n_estimators=100) #default value=100

regressor.fit(X_train,Y_train)
#evaluate model on test data
#predict to predict results


#saving model to disk
pickle.dump(regressor,open('model.pkl','wb'))
#for en
#test_data_prediction = regressor.predict(X_test)
#print(test_data_prediction)
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1351.949951 , 38.330002, 32.900002 , 1.324854]]))
#error_score = metrics.r2_score(Y_test,test_data_prediction)
#print('R squared error: ',error_score)