# Neural Network from Scratch on MNIST

This project implements a simple feedforward neural network from scratch (using only NumPy) to classify handwritten digits from the MNIST dataset. The implementation, training, and evaluation are all contained in a single Jupyter notebook: `main.ipynb`.

## Features
- **No deep learning frameworks**: The neural network is implemented using only NumPy for all computations.
- **MNIST dataset**: Uses the classic MNIST dataset of handwritten digits (0-9).
- **Single hidden layer**: The network consists of an input layer, one hidden layer (with ReLU activation), and an output layer (with softmax activation).
- **Training and evaluation**: Includes code for training, validation, and testing, with accuracy and loss metrics printed for each epoch.
- **Visualization**: Plots sample images and the confusion matrix for test predictions.

## Project Structure
- `main.ipynb`: Jupyter notebook containing all code for data loading, neural network implementation, training, evaluation, and visualization.

## Requirements
- Python 3.7+
- Jupyter Notebook
- NumPy
- Matplotlib
- scikit-learn
- keras (for loading MNIST dataset)

You can install the dependencies with:
```bash
pip install numpy matplotlib scikit-learn keras
```

## Usage
1. Clone this repository and navigate to the project directory.
2. Open `main.ipynb` in Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```
3. Run the notebook cells sequentially to:
   - Load and preprocess the MNIST data
   - Define and train the neural network
   - Evaluate performance on validation and test sets
   - Visualize results (sample images, confusion matrix)

## Results
- Achieves over **90% accuracy** on the MNIST test set after 150 epochs.
- Displays a confusion matrix to visualize classification performance across all digit classes.

## Output
```
Epoch 150/150 - loss: 0.8208 - accuracy: 0.9352 - val_loss: 1.4505 - val_accuracy: 0.9025
Train Accuracy: 0.9352
Validation Accuracy: 0.9025
Test Accuracy: 0.9044
```

## Confusion Matrix Example
A confusion matrix is plotted at the end of the notebook to show the distribution of predictions for each digit class.

## License
This project is for educational purposes and is provided as-is. 