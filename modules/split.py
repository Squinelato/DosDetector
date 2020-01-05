from sklearn.model_selection import StratifiedShuffleSplit

def splitting_train_test(dataset, test_size=0.2):
    
    X = dataset.drop(columns=['y']).values
    y = dataset.y
    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=42)
    sss.get_n_splits(X, y)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    return X_train, X_test, y_train, y_test