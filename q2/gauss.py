import numpy as np

def train_regularized(X, Y, lambda_reg):
  n_features, n_samples = X.shape
  n_classes = Y.shape[0]

  # Initialize containers
  priors = np.zeros(n_classes)
  means = np.zeros((n_classes, n_features))
  cov_matrices = []

  for c in range(n_classes):
    # Select samples belonging to class `c`
    indices = np.where(Y[c] == 1)[0]
    X_class = X[:, indices]

    # Calculate prior probability
    priors[c] = len(indices) / n_samples

    # Calculate class mean
    means[c] = np.mean(X_class, axis=1)

    # Calculate covariance matrix with regularization
    cov_matrix = np.cov(X_class, bias=True) + lambda_reg * np.identity(n_features)
    cov_matrices.append(cov_matrix)

  return priors, means, cov_matrices

def predict_regularized(X, priors, means, cov_matrices):
  n_samples = X.shape[1]
  n_classes = len(priors)
  log_posteriors = np.zeros((n_classes, n_samples))

  for c in range(n_classes):
    # Calculate log of prior
    log_prior = np.log(priors[c])

    # Calculate log likelihood
    mean_diff = X - means[c][:, np.newaxis]
    inv_cov = np.linalg.inv(cov_matrices[c])
    log_likelihood = -0.5 * np.sum((mean_diff.T @ inv_cov) * mean_diff.T, axis=1)
    log_det_cov = -0.5 * np.log(np.linalg.det(cov_matrices[c]))

    # Compute log-posterior
    log_posteriors[c] = log_prior + log_likelihood + log_det_cov

  # Predict class with highest log-posterior probability
  predictions = np.argmax(log_posteriors, axis=0) + 1
  return predictions

def evaluate_regularized(X_test, Y_test, priors, means, cov_matrices):
  predictions = predict_regularized(X_test, priors, means, cov_matrices)
  
  true_labels = np.argmax(Y_test, axis=0) + 1
  accuracy = np.mean(predictions == true_labels)
    
  return accuracy
