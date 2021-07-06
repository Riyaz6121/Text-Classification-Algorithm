import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

np.random.seed(0)

tv = TfidfVectorizer(stop_words='english', min_df=5, norm='l2')

data_train = pd.read_csv('training.csv')
feat_train = np.array(data_train.iloc[:,1])

X_train = tv.fit_transform(feat_train)
y_train = np.array(data_train.iloc[:,2])

data_test = pd.read_csv('test.csv')
feat_test = np.array(data_test.iloc[:,1])

X_test = tv.transform(feat_test)
y_test = np.array(data_test.iloc[:,2])

clf = LinearSVC(penalty = 'l2', dual=True, C=1, max_iter=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=clf.classes_))
