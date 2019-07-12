#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 17:46:24 2019

@author: albert24
"""

import pandas as pd
from sklearn import preprocessing
pd.set_option('display.max_columns', None)
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors, linear_model, tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns




class ProcessData:
    
    def cleanData(self):
        return
    
    def binFico(self, x):
        if x > 800:
            return "Exceptional"
        elif x > 740:
            return "Very Good"
        elif x > 670:
            return "Good"
        elif x > 580:
            return "Fair"
        else:
            return "Very Poor"
        
    def clean_emplength(self, x):
        if(x == 0):
            return "0"
        else:
            x = re.split('\+ | ', x)
            if (len(x) > 2):
                return x[1]
            else:
                return x[0]
                    
        
    
    def converttoCategorical(self, data):
        le = preprocessing.LabelEncoder()
        le.fit(data.fico_range_low)
        data['fico_range_low'] = le.transform(data.fico_range_low)
        
        le = preprocessing.LabelEncoder()
        le.fit(data.fico_range_high)
        data['fico_range_high'] = le.transform(data.fico_range_high)
        
        le = preprocessing.LabelEncoder()
        le.fit(data.last_fico_range_high)
        data['last_fico_range_high'] = le.transform(data.last_fico_range_high)
        
        le = preprocessing.LabelEncoder()
        le.fit(data.last_fico_range_low)
        data['last_fico_range_low'] = le.transform(data.last_fico_range_low)
        
        le = preprocessing.LabelEncoder()
        le.fit(data.verification_status)
        data['verification_status'] = le.transform(data.verification_status)
        
        le = preprocessing.LabelEncoder()
        le.fit(data.purpose)
        data['purpose'] = le.transform(data.purpose)
        
        le = preprocessing.LabelEncoder()
        le.fit(data.home_ownership)
        data['home_ownership'] = le.transform(data.home_ownership)
        
        le = preprocessing.LabelEncoder()
        le.fit(data.term)
        data['term'] = le.transform(data.term)
    
        
        le = preprocessing.LabelEncoder()
        le.fit(data.emp_length)
        data['emp_length'] = le.transform(data.emp_length)
        
        le = preprocessing.LabelEncoder()
        le.fit(data.addr_state)
        data['addr_state'] = le.transform(data.addr_state)
        
        le = preprocessing.LabelEncoder()
        le.fit(data.acc_now_delinq)
        data['acc_now_delinq'] = le.transform(data.acc_now_delinq)
        
        le = preprocessing.LabelEncoder()
        le.fit(data.delinq_2yrs)
        data['delinq_2yrs'] = le.transform(data.delinq_2yrs)
        
        return data
    
    def changeStatus(self, x):
        if (x == "Default"):
            return 1
        else:
            return 0
    def changelastcredit(self,x):
        
        x = re.split("[%s]" % ("".join("-")), str(x))
        #print(x)
        if(x[0]== 'Jan' or x[0]=='Feb' or x[0]=='Mar'):
            return "1"
        elif(x[0]=='Apr' or x[0]=='May' or x[0]=='Jun'):
            return"2"
        elif(x[0]=='Jul' or x[0]=='Aug' or x[0]=='Sep'):
            return "3"
        else:
            return"4"
        
    def selectColumns(self, data):
        data = data_df.drop(['issue_d', 'earliest_cr_line'], axis = 1)
        data['last_credit_pull_d'] = data.last_credit_pull_d.apply(lambda x: PD.changelastcredit(x))
        data['fico_range_low'] = data.fico_range_low.apply(lambda x: PD.binFico(x))
        data['fico_range_high'] = data.fico_range_high.apply(lambda x: PD.binFico(x))
        data['last_fico_range_high'] = data.last_fico_range_high.apply(lambda x: PD.binFico(x))
        data['last_fico_range_low'] = data.last_fico_range_low.apply(lambda x: PD.binFico(x))
        data['emp_length'] = data.emp_length.apply(lambda x: PD.clean_emplength(x))
        data['loan_status'] = data.loan_status.apply(lambda x: PD.changeStatus(x))
        return data
        
    def doNormalization(self, data):
        data = data.tolist()
        data = np.array(data)
        data = data.reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
#        normalizer = Normalizer(copy =True)
#        normalizer.fit(data)
#        data = normalizer.transform(data)
        return data
    
    def Normalization(self, data):
        data['annual_inc'] = PD.doNormalization(data['annual_inc'])
        data['loan_amnt'] = PD.doNormalization(data['loan_amnt'])
        data['installment'] = PD.doNormalization(data['installment'])
        data['dti'] = PD.doNormalization(data['dti'])
        data['delinq_amnt'] = PD.doNormalization(data['delinq_amnt'])
        data['mths_since_last_delinq'] = PD.doNormalization(data['mths_since_last_delinq'])
        data['mths_since_last_record'] = PD.doNormalization(data['mths_since_last_record'])
        data['inq_last_6mnths'] = PD.doNormalization(data['inq_last_6mths'])
        return data
    
    def plotcorrelationmatrix(self,data):
        f, ax = plt.subplots(figsize=(20, 12))
        corr = data.corr()
        hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                         linewidths=.05)
        f.subplots_adjust(top=0.93)
        t= f.suptitle('Correlation Matrix', fontsize=14)
        
    def plotconfusionmatrix(self,y_validate,y_predict,classifiername):
        classifierconf = confusion_matrix(y_validate,y_predict,labels = [0,1])
        classifierdata = pd.DataFrame((classifierconf.tolist()),index = ["On time","Default"],columns = ["On time","Default"])
        plt.figure(figsize = (5,5))
        plt.title("Classifier Confusion Matrix "+classifiername)
        sns.heatmap(classifierdata ,annot = True,fmt="d",cmap="YlGnBu")

    def Model(self,activatetest):
        models=[]
        models.append([tree.DecisionTreeClassifier(), {'min_samples_split': [2, 4, 6, 8, 10],#    |
                                                     'min_samples_leaf': [1, 5, 10, 15, 20],#     v
                                                      'max_depth': [5,10, 20, 30, 40, 50]},"Decision Tree"])
        
        models.append([linear_model.LogisticRegression(), {'penalty':['l1','l2'],'C': [0.001,0.01,0.1,1,10,100,1000,10000]}, "Logistic Regression"])
        
    
        models.append([neighbors.KNeighborsClassifier(), {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]},"KNeigbors"])
        
    #        
        models.append([RandomForestClassifier(), {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400,450,500,550]}, "Random Forest"])
        
        
        data_Y = data_train.loan_status
        data_X = data_train.drop(['loan_status'], axis = 1)
        
        
    
        
        X_train, X_validate, y_train, y_validate = train_test_split( data_X, data_Y, test_size=0.2, random_state=42)
        
        data_Y_test = data_test.loan_status
        data_X_test = data_test.drop(['loan_status'], axis = 1)
        
        #scores = cross_val_score(lr, data_X, data_Y, cv = 5, scoring = 'accuracy')
        for i in models:
            print(i[2])
            GS = GridSearchCV(i[0], i[1], cv = 3)
            GS.fit(X_train, y_train)
            
            #print(GS.best_estimator_)
            y_predict = GS.predict(X_validate)
            #y_prob = GS.predict_proba(X_validate)
            
            print(classification_report(y_validate, y_predict))
            PD.plotconfusionmatrix(y_validate, y_predict,i[2]+" validate")
            #print(y_prob)
            #print(roc_auc_score(data_Y_test, y_prob[:,1]))
      
            
                
            
            if(activatetest == True):
                
                y_predict_test = GS.predict(data_X_test)
                
                print("----------------------- TEST SET -------------------------")
                print(classification_report(data_Y_test,y_predict_test))
                PD.plotconfusionmatrix(data_Y_test,y_predict_test,i[2]+" test")
        
if __name__ == '__main__':
    
    loc = "data.csv"
    PD = ProcessData()
    data_df = pd.read_csv(loc)
    #print(data_df.last_credit_pull_d)
    data_df = data_df.fillna(0)
    data = PD.selectColumns(data_df)
    data = PD.converttoCategorical(data)
    
    
    print(data.groupby('loan_status')['loan_status'].count())
    data_default = data[data['loan_status']== 1]
    #print(data_default.count())
    data_paid = data[data['loan_status'] == 0]
    data_paid = data_paid.sample(n= 6037)
    #print(data_paid.count())
    
    
    data_default_train, data_default_test, data_paid_train, data_paid_test =  train_test_split(data_default,data_paid, test_size = 0.2, random_state = 42)
    #print(data_df.shape)
    #print(data_df.isna().any())
    #print(data_df.isnull().sum())
    #print(data_df['id'].nunique())
    
    data_train = pd.concat([data_default_train,data_paid_train])
    data_test = pd.concat([data_default_test,data_paid_test])
    data_train = PD.Normalization(data_train) 
    data_test = PD.Normalization(data_test)
    
    datatocsv = pd.concat([data_train,data_test])
#    print(datatocsv.term.unique())
#    print(datatocsv.emp_length.unique())
#    print(datatocsv.home_ownership.unique())
#    print(datatocsv.verification_status.unique())
#    print(datatocsv.loan_status.unique())
#    print(datatocsv.purpose.unique())
#    print(datatocsv.addr_state.unique())
#    print(datatocsv.fico_range_low.unique())
#    print(datatocsv.fico_range_high.unique())
#    print(datatocsv.acc_now_delinq.unique())
#    print(datatocsv.delinq_2yrs.unique())
#    print(datatocsv.last_credit_pull_d.unique())
    #print(datatocsv['loan_status'])
    PD.plotcorrelationmatrix(datatocsv)
    datatocsv.to_csv("Cleaneddata.csv")
    
    
    #change to true for running in test set
    PD.Model(True)
   
        
    
    
    
    
  
