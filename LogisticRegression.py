import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  ParameterGrid, StratifiedKFold
from sklearn.metrics import log_loss ,roc_auc_score,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
%config IPCompleter.greedy=True

DIR = '/'

def gini(y,pred):
    fpr,tpr,thr = roc_curve(y,pred,pos_label=1)
    g = 1 - 2 * auc(fpr,tpr)
    return g

if __name__ == '__main__':

df = pd.read_csv('train.csv')

x_train = df.drop('target',axis=1)
y_train = df['target'].values

use_cols = x_train.columns.values

cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=0)
all_params = {'C':[10**i for i in range(-1,2)],
              'fit_intercept':[True,False],
              'penalty':['l2','none'],
              'random_state':[0]}
min_score = 100
min_params = None

for params in tqdm(ParameterGrid(all_params)):
    list_gini_score = []
    list_logloss_score = []

    for train_idx, valid_idx in cv.split(x_train, y_train):
        train_x = x_train.iloc[train_idx,:]
        val_x = x_train.iloc[valid_idx,:]
    
        train_y = y_train[train_idx]
        val_y = y_train[valid_idx]
    
        clf = LogisticRegression(**params)
        clf.fit(train_x,train_y)
        pred = clf.predict(val_x)
        sc_gini = gini(val_y,pred)
        
        list_gini_score.append(sc_gini)
        
    sc_gini = np.mean(list_gini_score)
    
    if sc_gini < min_score:
        min_score = sc_gini
        min_params = params
        

clf = LogisticRegression(**min_params)
clf.fit(x_train,y_train)



df = pd.read_csv('test.csv')

x_test = df


pred_test = clf.predict(x_test)

df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE)

df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
df_submit['target']=pred_test
df_submit.to_csv(DIR+'submit.csv',index=False)