import numpy as np
import matplotlib.pyplot as plt

def load_data(plot=False):
  data = np.loadtxt('q2/EMGDataset.csv', delimiter=',')

  # Extract features and classes without transposing
  X = data[:2, :]  # X will be 2 x N (N = 50000), with p = 2 features
  features_x = data[0, :]  
  features_y = data[1, :]  
  classes = data[2, :]

  if plot: 
    class_labels = {
      1: "Neutro",
      2: "Sorriso",
      3: "Sobrancelhas Levantadas",
      4: "Surpreso",
      5: "Rabugento"
    }

    plt.figure(figsize=(10, 7))
    for class_value, label in class_labels.items():
      indices = classes == class_value
      plt.scatter(features_x[indices], features_y[indices], label=label, s=20, edgecolors='k', alpha=0.7)

    # Plot settings
    plt.title("EMG Data Scatter Plot")
    plt.xlabel("Sensor 1")
    plt.ylabel("Sensor 2")
    plt.legend(title="Classes", prop={'size': 12})
    plt.show()

  categories = data[2, :].astype(int)

  # Generate Y matrix with 5 classes for 50000 samples
  Y = -np.ones((5, categories.size))
  Y[categories - 1, np.arange(categories.size)] = 1 

  return X, Y
