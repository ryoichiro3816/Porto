import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  GridSearchCV, cross_val_score
from sklearn.metrics import log_loss ,roc_auc_score,roc_curve,auc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
%config IPCompleter.greedy=True

def gini(y,pred):
    fpr,tpr,thr = roc_curve(y,pred,pos_label=1)
    g = 1 - 2 * auc(fpr,tpr)
    return g

if __name__ == '__main__':

df = pd.read_csv('train.csv')

#データ型確認
df.dtypes

#特徴稜確認
df.describe() 

#欠損値確認
df.isnull().sum()

x_train = df.drop('target',axis=1)
y_train = df['target'].values

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
all_params = [{'svc_C':param_range, 'svc_kernel':['liner']},
              {'svc_C':param_range, 'svc_gamma':param_range,
               'svc_kernel':['rbf']}]
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=0))

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=all_params,
                  scoring='accuracy',
                  cv=2)
scores = cross_val_score(gs,
                         x_train,
                         y_train,
                         scoring='accuracy',
                         cv=5)
print('CV accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))


pipe_svc.fit(x_train,y_train)
y_pred = pipe_svc.predict(x_test)

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(2.5, 2.5))

ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for i in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
        
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.tight_layout()
plt.show()