from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import numpy as np

# Load the MNIST dataset and preprocess it
def load_dataset():
    # Load the data from the MNIST dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # Reshape data to include a single color channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # Convert labels to one-hot encoding format
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# Normalize pixel values
def prep_pixels(train, test):
    # Convert integer pixel values to floating-point numbers
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # Scale the values to the range [0, 1]
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

# Build and configure the CNN model
def define_model():
    model = Sequential()
    # Add a convolutional layer with 32 filters and ReLU activation
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    # Add max pooling to reduce the spatial dimensions
    model.add(MaxPooling2D((2, 2)))
    # Flatten the input before passing it to the fully connected layers
    model.add(Flatten())
    # Add a dense layer with 100 units and ReLU activation
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # Output layer with 10 units for each digit and softmax activation
    model.add(Dense(10, activation='softmax'))
    # Configure the optimizer and compile the model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Perform k-fold cross-validation to assess the model's performance
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = [], []
    # Initialize k-fold cross-validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # Loop over each split
    for train_ix, test_ix in kfold.split(dataX):
        # Initialize the CNN model
        model = define_model()
        # Split the data into training and testing sets for the current fold
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # Train the model on the current fold
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        # Evaluate the model on the validation set
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> Accuracy: %.3f' % (acc * 100.0))
        # Save the accuracy and training history
        scores.append(acc)
        histories.append(history)
    return scores, histories

# Plot training and validation curves for each fold
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # Plot the loss for training and validation
        plt.subplot(2, 1, 1)
        plt.title('Loss (Cross Entropy)')
        plt.plot(histories[i].history['loss'], color='blue', label='Train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='Validation')
        # Plot the accuracy for training and validation
        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='Train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='Validation')
    plt.show()

# Provide a summary of the model's performance
def summarize_performance(scores):
    # Print the mean and standard deviation of the model's accuracy
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # Create a box plot to visualize the accuracy scores
    plt.boxplot(scores)
    plt.show()

# Display a randomly chosen test sample along with its true and predicted labels
def display_sample_prediction(model, testX, testY):
    # Choose a random index from the test set
    idx = np.random.randint(0, testX.shape[0])
    # Extract the image and the true label
    image = testX[idx]
    true_label = np.argmax(testY[idx])
    
    # Reshape the image and make a prediction
    image = image.reshape((1, 28, 28, 1))  # Prepare the image for prediction
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    
    # Display the image along with the true and predicted labels
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'True Label: {true_label}, Predicted Label: {predicted_label}')
    plt.axis('off')
    plt.show()

# Orchestrate the entire process, from loading the dataset to evaluating the model
def run_test_harness():
    # Load and preprocess the dataset
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    # Evaluate the model using cross-validation
    scores, histories = evaluate_model(trainX, trainY)
    # Plot training and validation performance
    summarize_diagnostics(histories)
    # Provide a summary of the cross-validation results
    summarize_performance(scores)

    # Train the model on the entire training set
    model = define_model()
    model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
    
    # Display a prediction on a random sample from the test set
    display_sample_prediction(model, testX, testY)

# Entry point to execute the test harness
run_test_harness()
