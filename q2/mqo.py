import numpy as np
import matplotlib.pyplot as plt

def train(X, Y):
    # Adiciona uma coluna de 1s à esquerda de X
    X = np.c_[np.ones(X.shape[0]), X]
    
    # Modelo
    W = np.linalg.pinv(X.T @ X) @ X.T @ Y
    return W

def predict(X, W):
    X = np.c_[np.ones(X.shape[0]), X]
    return X @ W

def evaluate(X, Y, W):
    predictions = predict(X, W)
    
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(Y, axis=1)
    
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy



