import seaborn
import matplotlib
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(data, target, prediction, normalize='true'):
    seaborn.heatmap(confusion_matrix(data[target], data[prediction], normalize=normalize), annot=True)
    matplotlib.pyplot.xlabel('predict')
    matplotlib.pyplot.ylabel('target')