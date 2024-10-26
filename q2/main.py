import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import load_data
import mqo

X, Y = load_data.load_data(plot=False)
rounds = 500
models_acc = np.zeros((8,rounds))

for i in range(rounds):
  # train/test split
  num_samples = X.shape[0]
  indices = np.random.permutation(num_samples)
  split_idx = int(num_samples * 0.8)

  train_indices = indices[:split_idx]
  test_indices = indices[split_idx:]

  X_train, Y_train = X[train_indices], Y[train_indices]
  X_test, Y_test = X[test_indices], Y[test_indices]

  # Train the models
  W = mqo.train(X_train, Y_train)

  # Evaluate the models
  models_acc[0][i] = mqo.evaluate(X_test, Y_test, W)

  if i % 50 == 49:
    print(f"finished round {i+1}")

  
# Calculate statistics for each model
means = np.mean(models_acc, axis=1)
std_devs = np.std(models_acc, axis=1)
highest_values = np.max(models_acc, axis=1)
lowest_values = np.min(models_acc, axis=1)

# Create DataFrame for easier table display
models_names = ["MQO", "Gauss Trad", "Gauss Cov Iguais", "Gauss Cov Agregada", 'Bayes Ingenuo', "Gauss Reg 0,25", "Gauss Reg 0,50", "Gauss Reg 0,75"]
stats_table = pd.DataFrame({
    "Model": [models_names[i] for i in range(models_acc.shape[0])],
    "Mean Accuracy": means,
    "Standard Deviation": std_devs,
    "Highest Accuracy": highest_values,
    "Lowest Accuracy": lowest_values
})

print(stats_table)


bp=1