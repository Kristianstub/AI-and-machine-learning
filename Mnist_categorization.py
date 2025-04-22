import keras  # Import the Keras library


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # Load the MNIST dataset

    # Normalize the input data - MNIST data is pixel arrays, so divide by max pixel value 255
    x_train = x_train/255.0  # Normalize training data
    x_test = x_test/255.0  # Normalize test data

    # Output is categorical - map from digit target to vector (e.g. 2 -> [0,0,1,0,0,0,0,0,0,0])
    y_train = keras.utils.to_categorical(y_train, num_classes=10)  # Convert training labels to categorical
    y_test = keras.utils.to_categorical(y_test, num_classes=10)  # Convert test labels to categorical

    return x_train, y_train, x_test, y_test  # Return the processed data


def build_model(cnn):
    model = keras.Sequential()  # Initialize a sequential model

    # Input is 28x28 image, single channel (grayscale)
    model.add(keras.Input(shape=(28, 28, 1)))  # Add input layer

    if not cnn:
        ###  Fully connected neural network ###
        # Input is multidimensional, flattened to single dimension
        model.add(keras.layers.Flatten())  # Flatten the input
        # Add a hidden layer - units is number of neurons/layer width
        model.add(keras.layers.Dense(units=16, activation="relu"))  # Add dense layer with ReLU activation
        model.add(keras.layers.Dense(units=16, activation="relu"))
        model.add(keras.layers.Dense(units=16, activation="relu"))
        model.add(keras.layers.Dense(units=16, activation="relu"))
        # TODO add more dense layers and/or vary number of units for increased complexity of FNN

    else:
        ###  Convolutional neural network  ###
        # Add convolutional layer - filters is depth of layer output and kernel_size the convolution window
        model.add(keras.layers.Conv2D(filters=16, kernel_size=(10, 10), activation="relu", padding="same"))  # Add Conv2D layer
        # Add pooling layer to downscale (MaxPooling downscales by returning the maximum value in each input window)
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # Add MaxPooling2D layer
        # TODO add more layers and/or experiment with different number of filters, different kernel_size or pool_size

        # Flatten internal dimensions before output - additional dense layers could also be included after this line
        model.add(keras.layers.Flatten())  # Flatten the output

    # Final model layer - the same for all model architectures
    # Activation is softmax
   
    model.add(keras.layers.Dense(units=10, activation="softmax"))  # Add final dense layer with softmax activation

    return model  # Return the model


if __name__ == "__main__":
    # TODO try different values for epochs and learning_rate to improve model performance
    epochs = 10 # Set number of epochs
    learning_rate = 0.01  # Set learning rate

    x_train, y_train, x_test, y_test = load_mnist()  # Load and preprocess the MNIST data
    model = build_model(cnn=True)  # Build the model (set cnn=True for convolutional network, false for MLP)

    # Compile model - Stochastic gradient descent is chosen for the optimizer and categorical cross entropy for the
    # loss calculation
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)  # Initialize the SGD optimizer
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])  # Compile the model

    # Show model architecture details and compare parameter counts
    model.summary()  # Print model summary

    # Train the model on training data
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1, validation_split=0.1)  # Train the model

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)  # Evaluate the model
