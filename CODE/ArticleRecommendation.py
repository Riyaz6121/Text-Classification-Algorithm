import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

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

clf = LinearSVC(penalty='l2', loss='hinge', dual=True, max_iter=10)
clf.fit(X, y)

# Get a matrix of confidence values. Higher value means higher confidence
y_conf = clf.decision_function(test_X)

for j in range(len(clf.classes_)):

    cls = clf.classes_[j]
    
    # Get the articles with the 10 maximum confidence values for this class
    articles = np.argpartition(y_conf[:,j], -10)[-10:]
    
    # Remove any articles with confidence less than 0
    articles = [i for i in articles if y_conf[i, j] > 0]
    
    # Get the correct classifications for each recommended article
    real_classes = [y_test[article] for article in articles]
    
    # Standard metric calculation
    true_positive = real_classes.count(cls)
    false_positive = len(real_classes) - true_positive
    all_positive = len([c for c in y_test if c == cls])

    # If any are 0, all are 0.
    try:
        precision = true_positive/(true_positive+false_positive)
        recall = true_positive/all_positive
        f1 = 2 * (precision*recall)/(precision+recall)
    except ZeroDivisionError:
        precision = recall = f1 = 0
        
    print(cls)
    print('Recommended articles:')
    print(', '.join([str(y) for y in sorted([x+9501 for x in results[j]])]))
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')    
