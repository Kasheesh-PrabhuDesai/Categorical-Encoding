# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
""" 

import pandas as pd
from pandas import DataFrame



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

oh = OneHotEncoder(sparse=False)
lb = LabelEncoder()

def encoding(inp):
    inp['bin_3'] = lb.fit_transform(inp.bin_3.values.reshape(-1,1))
    inp['bin_4'] = lb.fit_transform(inp.bin_4.values.reshape(-1,1))
    inp['nom_0'] = oh.fit_transform(inp.nom_0.values.reshape(-1,1))
    inp['nom_1'] = oh.fit_transform(inp.nom_1.values.reshape(-1,1))
    inp['nom_2'] = oh.fit_transform(inp.nom_2.values.reshape(-1,1))
    inp['nom_3'] = oh.fit_transform(inp.nom_3.values.reshape(-1,1))
    inp['nom_4'] = oh.fit_transform(inp.nom_4.values.reshape(-1,1))
    inp['nom_5'] = oh.fit_transform(inp.nom_5.values.reshape(-1,1))
    inp['nom_6'] = oh.fit_transform(inp.nom_6.values.reshape(-1,1))
    inp['nom_7'] = oh.fit_transform(inp.nom_7.values.reshape(-1,1))
    inp['nom_8'] = oh.fit_transform(inp.nom_8.values.reshape(-1,1))
    inp['nom_9'] = lb.fit_transform(inp.nom_9.values.reshape(-1,1))
    #inp['nom_9'] = oh.fit_transform(inp.nom_9.values.reshape(-1,1))
    inp['ord_1'] = lb.fit_transform(inp.ord_1.values.reshape(-1,1))
    inp['ord_2'] = lb.fit_transform(inp.ord_2.values.reshape(-1,1))
    inp['ord_3'] = lb.fit_transform(inp.ord_3.values.reshape(-1,1))
    inp['ord_4'] = lb.fit_transform(inp.ord_4.values.reshape(-1,1))
    inp['ord_5'] = lb.fit_transform(inp.ord_5.values.reshape(-1,1))
    
    return inp

def training_phase(x,y,model):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=1)
    
    model = model.fit(x_train,y_train)
    
    return model,x_train,x_test,y_train,y_test

def testing_phase(x_test,y_test,model):
    
    result = model.predict(x_test)
    
    print(accuracy_score(y_test,result)*100)
    
    
def kaggle_submission(file):

    kaggle_set = pd.read_csv(file)
    
    id_ = kaggle_set['id']
    
    kaggle_set = kaggle_set.drop(['id','bin_0'],axis=1)
    
    kaggle_set = encoding(kaggle_set)
    
    return kaggle_set,id_
    
    
def main():
    
    
    df = pd.read_csv('train.csv')

    x = df.drop(['id','target','bin_0'],axis=1)
    y = df.iloc[:,24]
    
    model = XGBClassifier(learning_rate=0.1,n_estimators=300)

    
    
    x_set = encoding(x)
    
    model,x_train,x_test,y_train,y_test = training_phase(x_set,y,model)
    
    testing_phase(x_test,y_test,model)
    
    test_file,id_ = kaggle_submission('test.csv')
    
    prediction_df = DataFrame(model.predict_proba(test_file)[:,1],columns=['target'])
    
    id_df = DataFrame(id_,columns=['id'])
    
    final_df = pd.concat((id_df,prediction_df),axis=1)
    
    final_df.to_csv('final_df.csv')
    
    
    
    
    
    
    








