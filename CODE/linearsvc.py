import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.metrics import accuracy_score 

data = pd.read_csv('training.csv')
features = np.array(data.iloc[:,1])
y = np.array(data.iloc[:,2])
tv = TfidfVectorizer(stop_words='english', min_df=5, norm='l2')
X = tv.fit_transform(features)

#Code for gridsearchcv narrwing down the best parameter in LInearsvc() and timing.
s=time.time()
clf1 = LinearSVC(penalty = 'l2')
params = {'max_iter':[1, 5, 10],'loss':['squared_hinge', 'hinge'] ,'C':[10**-x for x in range(-5, 5)]}
grid = GridSearchCV(clf1, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)
print('The best parameters for clf1 is: ')
print(grid.best_params_)
print(e-s)
#----------------------------------------------

#CODE FOR ACCURACY TEST TO NARROW DOWN THE PARAMS FOR LINEARSVC (mainly done for max_iter)
c_vals = [100000,10000,1000,100,10,1,0.1,0.01,0.001,0.0001,0.00001]
max_iterat = [1,5,10]
acc_score_results=list()
for c in c_vals:
    for mi in max_iterat:
        model=LinearSVC(penalty = 'l2', dual=True, C=c, max_iter=mi)
        model.fit(X[:9000],y[:9000])
        y_pred=model.predict(X[9000:])
        score = accuracy_score(y[9000:], y_pred)
        acc_score_results.append(score)
print(acc_score_results)