# Script for training an a model for prediction
# on iris dataset

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

# Interactive mode for pyplot
plt.ion()

# Load iris dataset
df = pd.read_csv('../../Python Machine Learning/python-machine-learning-book/'
                 'code/datasets/iris/iris.data', header=None)

# Get feature vectors
X = df.iloc[:, 0:-1].values

# Get labels
y = df.iloc[:, -1].values

# Set integer classes
y[y == 'Iris-setosa'] = 0
y[y == 'Iris-versicolor'] = 1
y[y == 'Iris-virginica'] = 2

# Encode labels to one-hot encoding
y = np_utils.to_categorical(y, 3)

# Split the data into a training and a test set
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.33,
                                                random_state=0)

# Should probably normalize the features
# Instantiate sequential model
model = Sequential()

# First fully connected layer (hidden)
model.add(Dense(100, input_shape=(4,), name='HiddenLayer'))

# Add activation function
model.add(Activation('sigmoid', name='HiddenActivation'))

# Add output layer and activation
model.add(Dense(3, name='OutputLayer'))
model.add(Activation('softmax', name='OutputActivation'))

# Compile the model, and define optimizatoin criterion
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Train the model
modelSpecs = model.fit(XTrain, yTrain, batch_size=8, epochs=40,
                       validation_data=(XTest, yTest))

# Make a figure and add plot
fig = plt.figure()
ax = fig.add_subplot(111)
plot_labels = ['Validation loss', 'Validation accuracy', 'Training loss',
               'Training accuracy']
i = 0

# Plot training and validation error
for k in modelSpecs.history.keys():
    ax.plot(modelSpecs.history[k], label=plot_labels[i])
    i += 1

# Prettyfy plot
ax.set_title('Trainig of model')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy/Loss')
ax.legend()
ax.grid()

# Function for plotting both profiles on same graph
def save_figure(fig, filename):
    '''Saves the figure with desired filename.

    Figures get saved as .eps file with 1000 dpi.
    '''

    # Close all figures, because it messes with saving somehow
    plt.close('all')

    # Save figure
    fig.savefig(filename + '.eps', dpi=1000, format='eps', bbox_inches='tight')

# Save the figure
save_figure(fig, 'training')
