# MNIST Digit Recognition with Convolutional Neural Networks (CNN)

This project implements a Convolutional Neural Network (CNN) to perform handwritten digit recognition using the MNIST dataset. The model is built using TensorFlow and Keras, and it is evaluated using k-fold cross-validation to ensure robustness. The project includes preprocessing, model training, evaluation, and visualization of the results, as well as the ability to display predictions for individual samples.

## Project Workflow

1. **Dataset Loading and Preprocessing**:
   - The MNIST dataset is loaded and split into training and testing sets.
   - The images are reshaped to have a single channel (28x28x1) and pixel values are normalized to the range [0, 1].
   - The target labels are one-hot encoded for classification.

2. **Model Definition**:
   - A CNN model is defined with the following architecture:
     - **Conv2D**: 32 filters, 3x3 kernel size, ReLU activation
     - **MaxPooling2D**: 2x2 pooling
     - **Flatten**: Converts 2D feature maps to 1D vector
     - **Dense Layer**: 100 units, ReLU activation
     - **Output Layer**: 10 units (one for each digit), softmax activation
   - The model is compiled with Stochastic Gradient Descent (SGD) optimizer and categorical cross-entropy loss.

3. **Model Evaluation**:
   - The model is evaluated using 5-fold cross-validation.
   - For each fold, the model is trained for 10 epochs with a batch size of 32.
   - Accuracy is measured for each fold, and the mean and standard deviation of the scores are computed.

4. **Model Performance Summary**:
   - Learning curves for both cross-entropy loss and classification accuracy are plotted for each fold.
   - A box plot is used to summarize the accuracy scores across all folds.

5. **Sample Prediction**:
   - After training, a random test image is selected, and its true and predicted labels are displayed.
   - The image is visualized using Matplotlib, and the model's prediction is shown alongside the actual label.

## Key Functions

- `load_dataset()`: Loads and preprocesses the MNIST dataset.
- `prep_pixels(train, test)`: Normalizes pixel values of images.
- `define_model()`: Defines and compiles the CNN architecture.
- `evaluate_model(dataX, dataY, n_folds=5)`: Performs k-fold cross-validation to evaluate the model.
- `summarize_diagnostics(histories)`: Plots learning curves for each fold.
- `summarize_performance(scores)`: Summarizes the accuracy scores with a box plot.
- `display_sample_prediction(model, testX, testY)`: Displays a sample prediction from the test set.
- `run_test_harness()`: Runs the entire workflow from loading the dataset to evaluating the model and displaying predictions.

## Results

- **Cross-Validation Accuracy**: The model achieves competitive accuracy across multiple folds.
- **Sample Prediction**: Visualizations of true and predicted labels for randomly selected test samples.

## Visualizations

- Loss and accuracy learning curves.
- Box plot for cross-validation accuracy.
- Visualization of true vs. predicted labels on sample test images.

## Conclusion

This project demonstrates the effective use of a simple CNN architecture for handwritten digit recognition using the MNIST dataset. The use of k-fold cross-validation ensures reliable model performance across different splits of the data.
