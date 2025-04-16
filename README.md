# ICS425-Assignment4
# Exploring Fashion MNIST Dataset with Convolutional Neural Networks 

## Overview

This project explores the application of Convolutional Neural Networks (CNN) and how how different training configurations affect model performance and the predictions of an article of clothing in an image using the Tensorflow FashionMNIST dataset.

---

## Data Preprocessing

The following preprocessing steps were performed:

1. **Loading Data**  
   - The Fashion MNIST dataset is loaded and split into training, validation, and test sets.

2. **Reshaping**  
   - Grayscale images are reshaped to `(28, 28, 1)` to match the input shape expected by CNNs.

3. **Normalization**  
   - Pixel values are scaled to the range `[0, 1]` to stabilize and accelerate training.

4. **Data Augmentation** *(ModelAugmented only)*  
   - Performed using `ImageDataGenerator` to increase dataset variability.  
   - Augmentations include:
     - Rotation (±10 degrees)
     - Zoom (±10%)
     - Width and height shifts (±10%)
     - Horizontal flips

---

## Model Architecture

All models share the same CNN architecture and use the same optimizer:

- **Input → Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense (ReLU) → Dropout → Dense (Softmax)**
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Evaluation Metric:** Accuracy  

---
## Model Configurations

Four models were trained with the following differences:

1. **Baseline Model**
   - Epochs: `20`
   - Batch Size: `64`

2. **Model10**
   - Epochs: `10`
   - Batch Size: `64`

3. **Model32**
   - Epochs: `10`
   - Batch Size: `32`

4. **ModelAug**
   - Epochs: `20`
   - Batch Size: `64`
   - Uses data augmentation

---
## Evaluation

The script evaluates the models using the following metrics:

1. **Accuracy**
2. **Precision**
3. **Recall**
4. **F1-score**

It also includes a confusion matrix for each model and compares the results across the models using a bar graph.

## Ablation Study

This script performs an ablation study which varied:

- Number of training epochs
- Batch size
- Use of data augmentation

The goal was to identify how each hyperparameter influenced model performance.


## Requirements

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- scikit-learn  
- Matplotlib  

## Usage

1. Make sure you have the required libraries installed. You can install them using pip:
   ```bash pip install tensorflow numpy scikit-learn matplotlib```
3. Run the Python script in a Google Colab environment or Jupyter Notebook.
4. The script will train the models, evaluate their performance, and display the results.

## Author
Allison Ebsen
