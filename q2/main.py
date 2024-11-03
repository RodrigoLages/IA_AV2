import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import load_data
import mqo
import gauss

rounds = 500
X, Y = load_data.load_data(plot=False) # X.shape = (2, 50000), Y.shape = (5, 50000)
models_acc = np.zeros((8,rounds))

for i in range(rounds):
  # ____________________________ Data split ____________________________________
  num_samples = X.shape[1]  
  indices = np.random.permutation(num_samples)
  split_idx = int(num_samples * 0.8)

  train_indices = indices[:split_idx]
  test_indices = indices[split_idx:]

  X_train, Y_train = X[:, train_indices], Y[:, train_indices]
  X_test, Y_test = X[:, test_indices], Y[:, test_indices]

  # ____________________________ Training ____________________________________
  W_mqo = mqo.train(X_train, Y_train)
  means_shr, cov_matrix_shr = gauss.train_shared_covariance(X_train, Y_train)
  means_naive, cov_matrix_naive = gauss.train_naive_bayes(X_train, Y_train)
  pri_trad, means_trad, cov_matrices_trad = gauss.train_regularized(X_train, Y_train, lambda_reg=0)
  pri_025, means_025, cov_matrices_025 = gauss.train_regularized(X_train, Y_train, lambda_reg=0.25)
  pri_050, means_050, cov_matrices_050 = gauss.train_regularized(X_train, Y_train, lambda_reg=0.50)
  pri_075, means_075, cov_matrices_075 = gauss.train_regularized(X_train, Y_train, lambda_reg=0.75)
  pri_100, means_100, cov_matrices_100 = gauss.train_regularized(X_train, Y_train, lambda_reg=1)


  # ____________________________ Evaluating ____________________________________
  models_acc[0][i] = mqo.evaluate(X_test, Y_test, W_mqo)
  models_acc[2][i] = gauss.evaluate_linear_models(X_test, Y_test, means_shr, cov_matrix_shr)
  models_acc[4][i] = gauss.evaluate_linear_models(X_test, Y_test, means_naive, cov_matrix_naive)
  models_acc[1][i] = gauss.evaluate_regularized(X_test, Y_test, pri_trad, means_trad, cov_matrices_trad)
  models_acc[5][i] = gauss.evaluate_regularized(X_test, Y_test, pri_025, means_025, cov_matrices_025)
  models_acc[6][i] = gauss.evaluate_regularized(X_test, Y_test, pri_050, means_050, cov_matrices_050)
  models_acc[7][i] = gauss.evaluate_regularized(X_test, Y_test, pri_075, means_075, cov_matrices_075)
  models_acc[3][i] = gauss.evaluate_regularized(X_test, Y_test, pri_100, means_100, cov_matrices_100)


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