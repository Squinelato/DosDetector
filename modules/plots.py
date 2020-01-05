import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, fig, ax, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Rótulo Verdadeiro',
           xlabel='Rótulo Predito')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def confusionMatrix(classifiers, X_test_pre, y_test):
    
    cls_names = ['Linear', 'RBF', 'Sigmoid', 'Poly']
    
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    count = 0
    j = 0
    
    for i, cls in enumerate(classifiers):
              
        # predicting
        y_pred = cls.predict(X_test_pre)
        
        if count == 2:
            j = 1
        
        plot_confusion_matrix(y_test, y_pred, fig, axes[j][i % 2],
                              np.array(['Neutro', 'Positivo', 'Negativo']), 
                              normalize=True, title=cls_names[i])
        
        count += 1
        
    plt.savefig('Matriz/Todos.svg')    
    plt.show()
