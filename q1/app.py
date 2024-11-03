import numpy as np
from models_coefficients import get_mqo_coefficient
from models_coefficients import get_mqo_r_coefficients
from models_coefficients import get_mvo_coefficients
from models_coefficients import get_x_and_y
from plot_results import plot_results
import matplotlib.pyplot as plt
# Obter X_matrix e y_vector
data = get_x_and_y()
X_matrix = data['X_matrix']
y_vector = data['y_vector']

# Agora você pode rodar a simulação de Monte Carlo normalmente

# Função para calcular o RSS (Soma dos Resíduos Quadrados)
def calculate_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

# Número de rodadas da simulação
R = 500

# Listas para armazenar os resultados do RSS de cada modelo
rss_mqo = []
rss_mqo_r = {l: [] for l in [0, 0.25, 0.5, 0.75, 1]}
rss_mvo = []

# Início da simulação Monte Carlo
for i in range(R):
    if (i+1)%10 == 0: print(f"Rodadas: {i+1}")
    # Embaralhar os índices dos dados
    indices = np.random.permutation(len(X_matrix))
    
    # Dividir em 80% treino e 20% teste
    train_size = int(0.8 * len(X_matrix))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Criar conjuntos de treino e teste
    X_train, y_train = X_matrix[train_indices], y_vector[train_indices]
    X_test, y_test = X_matrix[test_indices], y_vector[test_indices]
    
    # Adicionar intercepto
    X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    # MQO Tradicional
    beta_mqo = get_mqo_coefficient(X_train_b, y_train)
    y_pred_mqo = X_test_b.dot(beta_mqo)
    rss_mqo.append(calculate_rss(y_test, y_pred_mqo))
    
    # MQO Regularizado (para cada lambda)
    for l in rss_mqo_r:
        beta_r = get_mqo_r_coefficients(X_train_b, y_train, l)
        y_pred_r = X_test_b.dot(beta_r)
        rss_mqo_r[l].append(calculate_rss(y_test, y_pred_r))
    
    # Média dos Valores Observáveis
    beta_mvo = get_mvo_coefficients(y_train)
    y_pred_mvo = np.full_like(y_test, beta_mvo[0])  # Média é o intercepto
    rss_mvo.append(calculate_rss(y_test, y_pred_mvo))

# Calcular estatísticas para cada modelo
def calculate_statistics(rss_values):
    mean_rss = np.mean(rss_values)
    std_rss = np.std(rss_values)
    max_rss = np.max(rss_values)
    min_rss = np.min(rss_values)
    return mean_rss, std_rss, max_rss, min_rss
# Função para formatar e imprimir as estatísticas
def print_statistics(stats_mqo, stats_mqo_r, stats_mvo):
    print("Estatísticas do MQO Tradicional (RSS):")
    print(f"  Média: {stats_mqo[0]:.4f}, Desvio Padrão: {stats_mqo[1]:.4f}, Máximo: {stats_mqo[2]:.4f}, Mínimo: {stats_mqo[3]:.4f}")

    print("\nEstatísticas do MQO Regularizado (RSS):")
    for l, stats in stats_mqo_r.items():
        print(f"  Lambda = {l}: Média: {stats[0]:.4f}, Desvio Padrão: {stats[1]:.4f}, Máximo: {stats[2]:.4f}, Mínimo: {stats[3]:.4f}")

    print("\nEstatísticas da Média dos Valores Observáveis (RSS):")
    print(f"  Média: {stats_mvo[0]:.4f}, Desvio Padrão: {stats_mvo[1]:.4f}, Máximo: {stats_mvo[2]:.4f}, Mínimo: {stats_mvo[3]:.4f}")

def get_regression_line(X, beta):
    # Calcula os valores preditos de y (linha de regressão) com base nos coeficientes beta
    # X aqui é a matriz com coluna de 1s (intercepto)
    y_pred = X.dot(beta)
    return y_pred

# Chamar a função com as estatísticas calculadas

# Calcular estatísticas para o MQO tradicional
stats_mqo = calculate_statistics(rss_mqo)

# Calcular estatísticas para MQO regularizado (para cada lambda)
stats_mqo_r = {l: calculate_statistics(rss_mqo_r[l]) for l in rss_mqo_r}

# Calcular estatísticas para a Média dos Valores Observáveis
stats_mvo = calculate_statistics(rss_mvo)

# Exibir os resultados
print_statistics(stats_mqo, stats_mqo_r, stats_mvo)
plot_results(stats_mqo, stats_mqo_r, rss_mqo_r, stats_mvo)
# Adicionar intercepto à matriz X para o cálculo da linha de regressão
X_b = np.hstack([np.ones((X_matrix.shape[0], 1)), X_matrix])

# Calcular os coeficientes beta usando o MQO (ou qualquer método desejado)
beta_mqo = get_mqo_coefficient(X_b, y_vector)

# Obter os valores preditos para traçar a linha
y_pred_line = get_regression_line(X_b, beta_mqo)

# Plotar o gráfico com os dados e a linha de regressão
plt.scatter(X_matrix, y_vector, label='Dados reais', color='blue')
plt.plot(X_matrix, y_pred_line, color='red', label='Linha de Regressão MQO')
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.legend()
plt.title('Gráfico de Dispersão com Linha de Regressão')
plt.show()

bp = 1