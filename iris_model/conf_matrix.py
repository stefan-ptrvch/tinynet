import itertools
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt


# Function for plotting both profiles on same graph
def save_figure(fig, filename):
    '''Saves the figure with desired filename.

    Figures get saved as .eps file with 1000 dpi.
    '''

    # Close all figures, because it messes with saving somehow
    plt.close('all')

    # Save figure
    fig.savefig(filename + '.eps', dpi=1000, format='eps', bbox_inches='tight')


# Function for making graphs out of confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.OrRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    mappable = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(mappable)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Set rotation of x ticks
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    return fig

# Turn on interactive mode for pyplot
plt.ion()

# Load model
model = load_model('./model_za_diplomski')

# Load dataset
iris = load_iris()

# Get class names
class_names = iris.target_names

# Predict on model
predictions = model.predict(iris['data'], batch_size=150)

# List for encoding predictions to labels
labels = np.array(150*[0])

# Put correct labels on predictions
labels[predictions[:, 0] == np.max(predictions, axis=1)] = 0

labels[predictions[:, 1] == np.max(predictions, axis=1)] = 1

labels[predictions[:, 2] == np.max(predictions, axis=1)] = 2

# Make confucion matrix
cnf_matrix = confusion_matrix(iris['target'], labels)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
fig0 = plot_confusion_matrix(cnf_matrix, classes=class_names,
                             title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
fig1 = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                             title='Normalized confusion matrix')

# Save figures
save_figure(fig0, 'conf_matrix_non_norm')
save_figure(fig1, 'conf_matrix_norm')
