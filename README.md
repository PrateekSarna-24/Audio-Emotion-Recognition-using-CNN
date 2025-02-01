# Audio Signal Classification Using Convolutional Neural Networks (CNNs)

## Overview

This project aims to classify different audio signals using Convolutional Neural Networks (CNNs). The goal is to build a model that can take audio signal data as input and predict its corresponding class.

## Directory Structure

The project folder contains the following sub-folders and files:

### 1. Data
The data folder contains:
- `audio_data` folder with raw audio files.
- Preprocessed audio data ready for training.
- Metadata files (e.g., class labels and descriptions).
- Audio feature extraction files (e.g., spectrograms, MFCC features).
- Saved model files and checkpoints.

### 2. Project Code Files
The project folder contains these main files:
- `CNN_Model.py`: Contains the functions for building, training, and evaluating the CNN model.
- `Feature_Extraction.py`: Extracts features like MFCCs from raw audio files.
- `Data_Preprocessing.py`: Handles the preprocessing of raw audio data and splits it into training/testing datasets.
- `Training_Evaluation.py`: Manages model training, evaluation, and accuracy reporting.

## Project Structure

### 1. Data Preprocessing & Feature Extraction
In this section, the following tasks were performed:
- Loaded audio signal data.
- Preprocessed audio files by converting them into a suitable format for model input (e.g., converting audio files into spectrograms or MFCCs).
- Split data into training and testing sets.
- Extracted features such as Mel-Frequency Cepstral Coefficients (MFCCs) to represent the audio signals in a numerical format.
- Normalized the features to ensure efficient model training.

### 2. Building the CNN Model
In this section, the following steps were followed:
- Defined the architecture of the CNN model using Keras/TensorFlow.
- The CNN consists of several convolutional layers, max-pooling layers, and fully connected layers for classification.
- Compiled the model with an appropriate optimizer and loss function.
- Trained the model using the prepared training dataset.
- Evaluated the model’s performance using the test dataset.
- Saved the trained model and weights.

#### Model Architecture

![Model Architecture](path_to_model_architecture_image)

### 3. Model Training & Evaluation
In this section, the following tasks were performed:
- Trained the CNN model for a set number of epochs.
- Evaluated the model's accuracy on the test dataset.
- Generated confusion matrices and classification reports.
- Saved the trained model for future use.
- Visualized the model’s performance (e.g., loss/accuracy plots).

#### Model Performance

![Model Performance](path_to_model_performance_image)

## Conclusion & Future Work

### Conclusion
- Built and trained a CNN model for audio signal classification.
- Achieved an accuracy of 85% on the test dataset.
- Successfully extracted meaningful features from raw audio files using MFCC and spectrograms.

### Future Work
- Implementing advanced architectures such as RNNs or hybrid models for better performance.
- Exploring larger audio datasets for better generalization.
- Incorporating data augmentation techniques like pitch shifting, time stretching, and noise addition.
- Developing a user-friendly interface for real-time audio classification using Streamlit or Flask.

### Submitted By
Prateek Sarna
