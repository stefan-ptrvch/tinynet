import itertools
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt

# Load model
model = load_model('./model_za_diplomski')

# Load dataset
iris = load_iris()

def saveModel(model):

    '''
    Function for saving the dimensions of the
    network, as well as the weights into
    a text file.
    '''

    # Calculate number of layers
    numLayers = len(model.get_weights()) // 2 + 1

    # Open text file for saving weights
    f = open("weights.txt", "w")

    # Write number of layers
    print('%d' % numLayers, end=' ', file=f)

    # Write layer sizes to file
    for i in range(0, numLayers, 2):

        # Write layer size
        layerSize = model.get_weights()[i].shape[0]
        print('%d' % layerSize, end=' ', file=f)

    # Wrie size of output layer
    layerSize = model.get_weights()[-1].shape[0]
    print('%d' % layerSize, end=' ', file=f)

    # Iterate over number of layers
    for i in range(0, numLayers, 2):

        # Convert bias to 2D array
        weights = model.get_weights()[i].T

        # Get weight matrix
        bias = model.get_weights()[i + 1][np.newaxis].T

        # Append bias to weight matrix
        weights = np.hstack((bias, weights))

        # Flatten matrix into list
        weights = weights.tolist()
        weights = [item for sublist in weights for item in sublist]

        # Write weights to file
        for item in weights:
            print('%f' % item, end=' ', file=f)

    # Close file
    f.close()

def save_model_new_lol(model):
    '''Saves weights and biases separately, according to new format. '''

    # Open text file for saving weights
    f = open("weights.h", "w")

    # Get weights
    weights = model.get_weights()

    # Calculate number of layers, not counting input layer
    num_layers = len(weights)

    # Go through layers
    for i in range(0, num_layers, 2):

        # Variable for formatting
        j = 0

        print("const float weight_matrix_%d[%d] = {\n" % (i//2, len(weights[i])
                                                                *len(weights[i][0])),
              end='', file=f)

        for weight_vector in weights[i].T:
            for weight in weight_vector:
                print("%f, " % weight, end='', file=f)
                j += 1
                if j == 7:
                    print("\n", end='', file=f)
                    j = 0


        print("};", file=f)

        j = 0

        print("const float bias_matrix_%d[%d] = {\n" % (i//2, len(weights[i + 1])),
              end='', file=f)

        for weight in weights[i + 1].T:
            print("%f, " % weight, end='', file=f)
            j += 1
            if j == 7:
                print("\n", end='', file=f)
                j = 0

        print("};", file=f)

    # Close file
    f.close()




def loadModel():

    '''
    Function for loading a model and setting up the
    invironment.
    '''

    from keras.models import Sequential
    import pandas as pd
    from keras.models import load_model
    # Load iris dataset
    df = pd.read_csv('~/Desktop/Machine Learning/Python Machine '
                     'Learning/python-machine-learning-book/code/'
                     'datasets/iris/iris.data', header=None)

    # Get feature vectors
    X = df.iloc[:, 0:-1].values

    # Load the model
    model = Sequential()
    model = load_model('irisKerasModel.h5')

    return model, X, df
