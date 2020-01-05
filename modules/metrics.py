from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from time import perf_counter
import pandas as pd
import numpy as np

def metrics(classifiers, X_test, y_test):
    
    results = list()
    columns_names = ['Acurácia balanceada', 'Precisão', 'Recall', 'F1', 'Tempo', 'Desvio Tempo']
    index_names   = ['Linear', 'RBF', 'Sigmoid', 'Poly']
    
    for cls in classifiers:
        
        y_pred = None
        exe_time = list()
        
        # predicting
        for i in range(30):
            start_time = perf_counter()
            y_pred = cls.predict(X_test)
            elapsed_time = perf_counter() - start_time
            exe_time.append(elapsed_time)
        
        # generating the metrics
        acc = balanced_accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1s = f1_score(y_test, y_pred)
        
        results.append([acc, pre, rec, f1s, round(np.mean(exe_time), 4), 
                                            round(np.std(exe_time), 4)])
    
    return pd.DataFrame(results, index=index_names, columns=columns_names)

def timeMesure(classifiers, X_test, y_test):
    
    results = list()
    
    for cls in classifiers:
        
        y_pred = None
        exe_time = list()
        
        # predicting
        for i in range(30):
            start_time = perf_counter()
            y_pred = cls.predict(X_test)
            elapsed_time = perf_counter() - start_time
            exe_time.append(elapsed_time)
            
        results.append(exe_time)
        
    return results