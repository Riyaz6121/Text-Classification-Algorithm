CODE FOR COMPARING LINEARSVC() WITH OTHER SVC MODEL WITH DIFFERENT KERNELS

data = pd.read_csv('training.csv')
features = np.array(data.iloc[:,1])
y = np.array(data.iloc[:,2])

# define the models and feature extractions we'll be trying

#add stuff on why you decided to add in these parameters
vectorizers = [TfidfVectorizer(stop_words='english', min_df=5, norm='l2')]
models = [LinearSVC(),
          SVC(kernel='poly'),
          SVC(kernel='sigmoid'),
          SVC(kernel='rbf')]

results = []

# test each model on each vectorizer
for vectorizer in vectorizers:
    X = vectorizer.fit_transform(features)
    for model in models:
        skf = StratifiedKFold(n_splits=5)
        for train_i, test_i in skf.split(X, y):
            model.fit(X[train_i], y[train_i])
            results.append(model.score(X[test_i], y[test_i]))
print(len(results))
print(results)