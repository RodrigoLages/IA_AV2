import numpy as np

def train_regularized(X, Y, lambda_reg):
  n_features, n_samples = X.shape
  n_classes = Y.shape[0]

  priors = np.zeros(n_classes)
  means = np.zeros((n_classes, n_features))
  cov_matrices = []

  for c in range(n_classes):
    indices = np.where(Y[c] == 1)[0]
    X_class = X[:, indices]

    priors[c] = len(indices) / n_samples

    means[c] = np.mean(X_class, axis=1)

    cov_matrix = np.cov(X_class, bias=True) + lambda_reg * np.identity(n_features)
    cov_matrices.append(cov_matrix)

  return priors, means, cov_matrices

def predict_regularized(X, priors, means, cov_matrices):
  n_samples = X.shape[1]
  n_classes = len(priors)
  log_posteriors = np.zeros((n_classes, n_samples))

  for c in range(n_classes):
    log_prior = np.log(priors[c])

    mean_diff = X - means[c][:, np.newaxis]
    inv_cov = np.linalg.pinv(cov_matrices[c])
    log_likelihood = -0.5 * np.sum(mean_diff * (inv_cov @ mean_diff), axis=0)
    log_det_cov = -0.5 * np.log(np.linalg.det(cov_matrices[c]))

    log_posteriors[c] = log_prior + log_likelihood + log_det_cov

  predictions = np.argmax(log_posteriors, axis=0) + 1
  return predictions

def evaluate_regularized(X_test, Y_test, priors, means, cov_matrices):
  predictions = predict_regularized(X_test, priors, means, cov_matrices)
  
  true_labels = np.argmax(Y_test, axis=0) + 1
  accuracy = np.mean(predictions == true_labels)
    
  return accuracy

def train_shared_covariance(X, Y):
  n_features = X.shape[0]
  n_classes = Y.shape[0]

  means = np.zeros((n_classes, n_features))
  #means = np.array([X[:, Y[c] == 1].mean(axis=1) for c in range(num_classes)]).T 
  for c in range(n_classes):
    indices = np.where(Y[c] == 1)[0]
    X_class = X[:, indices]
    means[c] = np.mean(X_class, axis=1)
  
  cov_matrix = np.cov(X, bias=True) 

  return means, cov_matrix

def train_naive_bayes(X, Y):
  n_features = X.shape[0]
  n_classes = Y.shape[0]

  means = np.zeros((n_classes, n_features))
  #means = np.array([X[:, Y[c] == 1].mean(axis=1) for c in range(num_classes)]).T 
  for c in range(n_classes):
    indices = np.where(Y[c] == 1)[0]
    X_class = X[:, indices]
    means[c] = np.mean(X_class, axis=1)
  
  variances = np.diag(np.var(X, axis=1))

  return means, variances

def predict_linear_models(X, means, cov_matrix):
    n_samples = X.shape[1]  
    n_classes = means.shape[0]  
    distances = np.zeros((n_classes, n_samples))  
    inv_cov = np.linalg.pinv(cov_matrix) 
    
    for c in range(n_classes):
        
        mean_diff = X - means[c][:, np.newaxis]  
        distance = np.sum(mean_diff * (inv_cov @ mean_diff), axis=0)

        distances[c] = distance  

    predictions = np.argmin(distances, axis=0)  
    
    return predictions

def evaluate_linear_models(X_test, Y_test, means, cov_matrix):
  predictions = predict_linear_models(X_test, means, cov_matrix)
  
  true_labels = np.argmax(Y_test, axis=0)
  accuracy = np.mean(predictions == true_labels)
    
  return accuracy



