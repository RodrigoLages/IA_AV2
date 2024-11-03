import matplotlib.pyplot as plt
import numpy as np
def plot_results(stats_mqo, stats_mqo_r, rss_mqo_r, stats_mvo):
    # Extrair os dados das estatísticas
    labels = ['MQO Tradicional', 'MQO Regularizado λ=0', 'MQO Regularizado λ=0.25',
            'MQO Regularizado λ=0.5', 'MQO Regularizado λ=0.75', 'MQO Regularizado λ=1', 
            'Média dos Valores Observáveis']

    # Organizar as estatísticas em listas
    mean_values = [stats_mqo[0]] + [stats_mqo_r[l][0] for l in rss_mqo_r] + [stats_mvo[0]]
    std_values = [stats_mqo[1]] + [stats_mqo_r[l][1] for l in rss_mqo_r] + [stats_mvo[1]]

    # Definindo a largura das barras
    bar_width = 0.35

    # Criando o gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    # Barras para média
    bars1 = ax.bar(np.arange(len(labels)), mean_values, bar_width, label='Média RSS', alpha=0.7)

    # Barras para desvio padrão
    bars2 = ax.bar(np.arange(len(labels)) + bar_width, std_values, bar_width, label='Desvio Padrão RSS', alpha=0.7)

    # Adicionando detalhes ao gráfico
    ax.set_xlabel('Modelos')
    ax.set_ylabel('Valores de RSS')
    ax.set_title('Desempenho dos Modelos de Regressão')
    ax.set_xticks(np.arange(len(labels)) + bar_width / 2)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(axis='y')

    # Exibindo o gráfico
    plt.tight_layout()
    plt.show()