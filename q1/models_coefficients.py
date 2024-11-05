import numpy as np
import matplotlib.pyplot as plt


dados = np.loadtxt('q1/aerogerador.dat')


X = dados[:, 0]  
y = dados[:, 1]  


plt.scatter(X, y)
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.title('Gráfico de Dispersão: Velocidade do Vento vs Potência Gerada')


X_matrix = X.reshape(-1, 1)  
y_vector = y.reshape(-1, 1)  
X_b = np.hstack([np.ones((X_matrix.shape[0], 1)), X_matrix]) 

def get_x_and_y():
    return {"X_matrix" : X_matrix, "y_vector": y_vector}
#{----------------------------- MQO tradicional  -----------------------------}
def get_mqo_coefficient(X, y):
    
    beta_mqo = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return beta_mqo.ravel()

#{----------------------------- MQO regularizado -----------------------------}
def get_mqo_r_coefficients(X, y, l):
    
    beta_mqo_r = np.linalg.pinv(X.T.dot(X) + l * np.eye(X.shape[1])).dot(X.T).dot(y)
    return beta_mqo_r.ravel()

#{----------------------- Média dos valores observáveis ----------------------}
def get_mvo_coefficients(y):
    
    media_y = np.mean(y)
    return np.array([media_y])  
    