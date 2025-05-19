# Student Loan Risk Prediction with Deep Learning

## Overview
This project implements a deep learning model to predict student loan risk based on various student characteristics. The model is built using TensorFlow and Keras, and it demonstrates the application of neural networks in financial risk assessment for student loans.

## Features
- Data preprocessing and feature engineering for student loan data
- Implementation of sequential neural network models using TensorFlow/Keras
- Model training with different architectures and hyperparameters
- Performance evaluation using classification metrics
- Visualization of training history and model performance

## Dataset
The dataset includes various student attributes such as:
- Payment history
- Location parameters
- STEM degree scores
- GPA rankings
- Alumni success metrics
- Study major codes
- Time to completion
- Financial workshop scores
- And more...

## Technologies Used
- Python 3.x
- TensorFlow 2.x
- Keras
- Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Lambourne2/neural-network-challenge-1.git
   cd neural-network-challenge-1
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook student_loans_with_deep_learning.ipynb
   ```

2. Run the notebook cells sequentially to:
   - Load and preprocess the data
   - Split the data into training and testing sets
   - Build and train the neural network models
   - Evaluate model performance
   - Visualize the results

## Model Architecture
The project implements two neural network models:
1. **Baseline Model**:
   - Input layer with 12 features
   - Hidden layer with 80 neurons (ReLU activation)
   - Output layer with 1 neuron (Sigmoid activation)

2. **Optimized Model**:
   - Input layer with 12 features
   - First hidden layer with 100 neurons (ReLU activation)
   - Second hidden layer with 80 neurons (ReLU activation)
   - Third hidden layer with 50 neurons (ReLU activation)
   - Output layer with 1 neuron (Sigmoid activation)

## Results
The models are evaluated based on:
- Accuracy
- Loss
- Classification report (precision, recall, f1-score)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Data provided by [data source]
- Built as part of a machine learning challenge