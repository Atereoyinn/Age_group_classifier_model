# Age Classification from Images

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify images into three age categories: YOUNG, MIDDLE, and OLD. The model is trained on a dataset of facial images and uses data augmentation techniques to improve performance.

## Dataset

The dataset consists of images categorized into three age groups. The data is loaded from a CSV file containing image IDs and their corresponding age class labels.

## Project Structure

The project is implemented as a Jupyter notebook containing all the analysis and model training steps.

## Dependencies

- pandas
- matplotlib
- scikit-learn
- numpy
- tensorflow
- seaborn
- PIL
- anvil-uplink

## Data Preprocessing

1. Load data from CSV file
2. Encode age classes to numerical values (0: YOUNG, 1: MIDDLE, 2: OLD)
3. Visualize class distribution
4. Implement custom image reading and resizing functions
5. Create image paths and corresponding labels
6. Implement data augmentation using ImageDataGenerator

## Model Architecture

The CNN model consists of:
- Two convolutional layers with max pooling
- Flatten layer
- Dense layer with dropout
- Output layer with softmax activation

## Training

- Data split: 80% training, 20% testing
- Batch size: 64
- Epochs: 15
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy

## Evaluation

- Accuracy and loss plots for both original and augmented models
- Confusion matrix
- Classification report
- ROC curve analysis

## Results

- The model achieves an accuracy of approximately 85% on the test set
- Augmented model shows improved performance in handling overfitting

## Deployment

The model is deployed using Anvil, allowing for real-time image classification through a web interface.

## Usage

To use the deployed model:

1. Connect to the Anvil server using the provided uplink key
2. Use the `classify_image` function to predict the age category of a given image

## Future Work

- Fine-tune model architecture for improved performance
- Experiment with different data augmentation techniques
- Implement cross-validation for more robust evaluation
- Explore transfer learning with pre-trained models

## Contact

[Abiola Ayuba] - [abiolaayubam@gmail.com]

Project Link: [https://github.com/atereoyinn/Age_group_classifier_model]

UI Link: [https://7piz3aksd4kp2tc2.anvil.app/VG3ZUQC6RJU5TUDV6SU6NPZ2]
