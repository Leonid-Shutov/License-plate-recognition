import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Input data - a grayscale image of size 28x28. 1 means using of only one gray channel. 
input_shape = (28, 28, 1)
# Output size is a number of classes we use. For MNIST, it's 10 (10 digits to classify).
output_size = 10

# MNIST DATASET
# Here we load MNIST data and split it into 2 groups: train and test.
# The data is represented as a pair of X and Y - input and expected output.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# Original MNIST data consists of values in the [0, 255] range (alpha channel).
# Here we scale it to the [0, 1] range by diviving to 255.
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure the input data items have shape (28, 28, 1).
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

test = x_train[0]

print("x_train shape:", x_train[0].shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Original output values of the MNIST dataset are provided as class labels:
# 0 for "0" image, 1 for "1", and so on.
# Here we convert it to a more convenient format:
# 0 - (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# 1 - (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
# 2 - (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
# ...
# 9 - (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
y_train = keras.utils.to_categorical(y_train, output_size)
y_test = keras.utils.to_categorical(y_test, output_size)

# Let's build the model now.
# Sequential means that the layers in a model go one by one.
# In other words, a pretty simple neural network.
model = keras.Sequential(
    [
        # Input: 28x28x1
        layers.InputLayer(input_shape=input_shape),
        # The 1st convolution: 6 kernels of size 3x3
        layers.Conv2D(6, kernel_size=(3, 3), activation="relu"),
        # Max pooling: we take a window of 3x3, select a max value from it
        # and set in a result pixel. The step vector (stride) is 2x2.
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # The 2nd convolution: 16 kernels of size 3x3.
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        # Another one max pooling.
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # Flatten is required to convert a multidimensional layer
        # (convolution or pooling is 3-dimensional since we have
        # N feature maps of WxH size).
        layers.Flatten(),
        # A fully connected layer of 120 neurons.
        layers.Dense(120, activation="relu"),
        # The last, output layer. Activation is Softmax - we will learn
        # about it in future.
        layers.Dense(output_size, activation="softmax"),
    ]
)

# Just prints the model info.
model.summary()

# Here we go for training.
# A little reminder - training is a process of setting a model's weights ("knowledge").
# Techniques used for training are backpropagation and stochastic gradient descent (SGD).

# Batch size is a number of items we take from a dataset to show the model.
# Here, we show 32 items and then update the model's weights.
batch_size = 32

# Epoch is another hyperparameter that shows how long we will train the model.
# An epoch ends when all data from the dataset is shown to the network.
# Let's summarize: if we have a dataset of 200 items, and the batch size is 32,
# then, to pass through one epoch we show 200/32 = 6.25 -> 7 batches to the model
# (6 full and 1 imcomplete). Then another epoch begins.
# A little note - the MNIST dataset has 60000 train and 10000 test items.
epochs = 30

# Here we compile the model for training.
model.compile(
    # Crossentropy is a type of a loss commonly used for classification training.
    loss=keras.losses.categorical_crossentropy,
    # SGD - our well known stochastic gradient descent.
    # Actually, here we have a Momentum SGD - an upgrade to classic SGD that
    # takes into account the results of the previous weights update.
    # We will talk about other optimizers on the next lecture.
    optimizer=keras.optimizers.SGD(0.01, 0.9),
    # Accuracy - a readable metric to see how the training is going.
    metrics=["accuracy"])

# Here we launch the training process.
# We pass X and Y items of the train dataset, set the batch size and epoch count.
# Validation split is a percentage of the dataset used for validation during training.
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# print('predict', model.predict(test))
model.save('mnist_32_30.h5')

# After training, we evaluate the model on the test dataset.
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])