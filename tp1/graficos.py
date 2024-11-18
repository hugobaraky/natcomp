import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import numpy as np

# Configuração do estilo dos gráficos
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})  # Evita avisos de muitas figuras abertas

# Diretório onde os arquivos de log estão armazenados
log_directory = 'logs/'

# Padrão de nomenclatura dos arquivos de log (ajuste conforme necessário)
# Exemplo: log_pop100_gen30.csv
log_pattern = os.path.join(log_directory, 'log_pop*_gen*.csv')

# Função para extrair parâmetros do nome do arquivo usando regex
def extract_parameters(filename):
    """
    Extrai os parâmetros de interesse a partir do nome do arquivo.

    Args:
        filename (str): Nome do arquivo.

    Returns:
        dict: Dicionário com os parâmetros extraídos.
    """
    pattern = r'log_pop(?P<population>\d+)_gen(?P<generations>\d+)\.csv'
    match = re.search(pattern, filename)
    if match:
        return {
            'Population Size': int(match.group('population')),
            'Generations': int(match.group('generations'))
        }
    else:
        # Adicione outros parâmetros conforme necessário
        return {}

# 1. Leitura e Consolidação dos Logs
log_files = glob.glob(log_pattern)

# Lista para armazenar os DataFrames de cada log com parâmetros
log_data_list = []

for file in log_files:
    params = extract_parameters(os.path.basename(file))
    if not params:
        print(f"Parâmetros não encontrados no arquivo: {file}")
        continue
    try:
        df = pd.read_csv(file)
        # Adicionar colunas de parâmetros
        for key, value in params.items():
            df[key] = value
        log_data_list.append(df)
    except pd.errors.ParserError as e:
        print(f"Erro ao processar o arquivo {file}: {e}")
    except Exception as e:
        print(f"Erro inesperado ao processar o arquivo {file}: {e}")

# Concatenar todos os DataFrames em um único DataFrame
if log_data_list:
    consolidated_log_df = pd.concat(log_data_list, ignore_index=True)
    print("Logs consolidados com sucesso.")
else:
    print("Nenhum arquivo de log encontrado ou nenhum parâmetro extraído.")
    consolidated_log_df = pd.DataFrame()

# 2. Análise e Agregação dos Dados

# Verificar se o DataFrame não está vazio
if not consolidated_log_df.empty:
    # Verificar se as colunas necessárias existem
    required_columns = [
        'Population Size', 'Generations', 'Average Fitness',
        'Max Fitness', 'Min Fitness', 'Generation Time (s)',
        'Train V-measure', 'Test V-measure', 'Generation'
    ]
    missing_columns = [col for col in required_columns if col not in consolidated_log_df.columns]
    if missing_columns:
        print(f"As seguintes colunas estão faltando nos dados consolidados: {missing_columns}")
        complete_summary = pd.DataFrame()
    else:
        try:
            # Calcular estatísticas agregadas por parâmetros
            aggregated_stats = consolidated_log_df.groupby(['Population Size', 'Generations']).agg({
                'Average Fitness': ['mean', 'std'],
                'Max Fitness': ['mean', 'std'],
                'Min Fitness': ['mean', 'std'],
                'Generation Time (s)': ['mean', 'std']
            }).reset_index()
            
            # Flatten dos nomes das colunas
            aggregated_stats.columns = ['Population Size', 'Generations',
                                        'Avg_Fitness_Mean', 'Avg_Fitness_STD',
                                        'Max_Fitness_Mean', 'Max_Fitness_STD',
                                        'Min_Fitness_Mean', 'Min_Fitness_STD',
                                        'Gen_Time_Mean', 'Gen_Time_STD']
            
            # Calcular estatísticas finais (média e desvio padrão) por conjunto de parâmetros
            final_summary = consolidated_log_df.groupby(['Population Size', 'Generations']).agg({
                'Train V-measure': ['mean', 'std'],
                'Test V-measure': ['mean', 'std']
            }).reset_index()
            
            final_summary.columns = ['Population Size', 'Generations',
                                     'Train_V_Measure_Mean', 'Train_V_Measure_STD',
                                     'Test_V_Measure_Mean', 'Test_V_Measure_STD']
            
            # Mesclar as estatísticas agregadas com o resumo final
            complete_summary = pd.merge(aggregated_stats, final_summary,
                                        on=['Population Size', 'Generations'])
        except KeyError as e:
            print(f"Erro ao agregar dados: {e}")
            complete_summary = pd.DataFrame()
else:
    complete_summary = pd.DataFrame()

# 3. Geração de Gráficos Comparativos

# Função para plotar a evolução do fitness
def plot_fitness_evolution(consolidated_log_df, save_path):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=consolidated_log_df, x='Generation', y='Average Fitness',
                 hue='Population Size', style='Generations', ci='sd')
    plt.xlabel('Geração', fontsize=14)
    plt.ylabel('Fitness Médio', fontsize=14)
    plt.title('Evolução do Fitness Médio ao Longo das Gerações', fontsize=16)
    plt.legend(title='População / Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Alterado para close para evitar exibição interativa

# Função para plotar a distribuição da Fitness Final
def plot_final_fitness_distribution(consolidated_log_df, save_path):
    # Filtrar a última geração
    max_generation = consolidated_log_df['Generation'].max()
    final_gen_df = consolidated_log_df[consolidated_log_df['Generation'] == max_generation]
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=final_gen_df, x='Population Size', y='Max Fitness', hue='Generations')
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Fitness Máximo na Última Geração', fontsize=14)
    plt.title('Distribuição da Fitness Máxima na Última Geração', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure
def plot_v_measure_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Train_V_Measure_Mean',
                hue='Generations', palette='Set2', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Train_V_Measure_Mean'],
                     yerr=subset['Train_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Treino', fontsize=14)
    plt.title('Comparação da V-measure Média no Treino por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure no Teste
def plot_v_measure_test_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Test_V_Measure_Mean',
                hue='Generations', palette='Set1', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Test_V_Measure_Mean'],
                     yerr=subset['Test_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Teste', fontsize=14)
    plt.title('Comparação da V-measure Média no Teste por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure no Teste
def plot_v_measure_test_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Test_V_Measure_Mean',
                hue='Generations', palette='Set1', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Test_V_Measure_Mean'],
                     yerr=subset['Test_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Teste', fontsize=14)
    plt.title('Comparação da V-measure Média no Teste por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure no Teste
def plot_v_measure_test_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Test_V_Measure_Mean',
                hue='Generations', palette='Set1', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Test_V_Measure_Mean'],
                     yerr=subset['Test_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Teste', fontsize=14)
    plt.title('Comparação da V-measure Média no Teste por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure no Teste
def plot_v_measure_test_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Test_V_Measure_Mean',
                hue='Generations', palette='Set1', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Test_V_Measure_Mean'],
                     yerr=subset['Test_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Teste', fontsize=14)
    plt.title('Comparação da V-measure Média no Teste por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure no Teste
def plot_v_measure_test_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Test_V_Measure_Mean',
                hue='Generations', palette='Set1', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Test_V_Measure_Mean'],
                     yerr=subset['Test_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Teste', fontsize=14)
    plt.title('Comparação da V-measure Média no Teste por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure no Treino
def plot_v_measure_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Train_V_Measure_Mean',
                hue='Generations', palette='Set2', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Train_V_Measure_Mean'],
                     yerr=subset['Train_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Treino', fontsize=14)
    plt.title('Comparação da V-measure Média no Treino por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure no Teste
def plot_v_measure_test_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Test_V_Measure_Mean',
                hue='Generations', palette='Set1', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Test_V_Measure_Mean'],
                     yerr=subset['Test_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Teste', fontsize=14)
    plt.title('Comparação da V-measure Média no Teste por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure no Teste
def plot_v_measure_test_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Test_V_Measure_Mean',
                hue='Generations', palette='Set1', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Test_V_Measure_Mean'],
                     yerr=subset['Test_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Teste', fontsize=14)
    plt.title('Comparação da V-measure Média no Teste por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure no Teste
def plot_v_measure_test_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Test_V_Measure_Mean',
                hue='Generations', palette='Set1', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Test_V_Measure_Mean'],
                     yerr=subset['Test_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Teste', fontsize=14)
    plt.title('Comparação da V-measure Média no Teste por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure no Treino
def plot_v_measure_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Train_V_Measure_Mean',
                hue='Generations', palette='Set2', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Train_V_Measure_Mean'],
                     yerr=subset['Train_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Treino', fontsize=14)
    plt.title('Comparação da V-measure Média no Treino por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para plotar a comparação da V-measure no Teste
def plot_v_measure_test_comparison(complete_summary, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=complete_summary, x='Population Size', y='Test_V_Measure_Mean',
                hue='Generations', palette='Set1', capsize=0.1)
    # Ajustar posições para errorbar
    population_sizes = complete_summary['Population Size'].unique()
    generations = complete_summary['Generations'].unique()
    num_pops = len(population_sizes)
    num_gens = len(generations)
    bar_width = 0.8 / num_gens  # Distribuir barras para diferentes gerações
    
    for i, gen in enumerate(generations):
        subset = complete_summary[complete_summary['Generations'] == gen]
        indices = np.arange(num_pops) + i * bar_width
        plt.errorbar(indices, subset['Test_V_Measure_Mean'],
                     yerr=subset['Test_V_Measure_STD'],
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Média da V-measure no Teste', fontsize=14)
    plt.title('Comparação da V-measure Média no Teste por Parâmetro', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Função para analisar o impacto de cada parâmetro individualmente
def plot_parameter_impact(complete_summary, parameter, metric, save_path):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=complete_summary, x=parameter, y=metric,
                 hue='Generations', marker='o')
    plt.xlabel(parameter, fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(f'Impacto de {parameter} na {metric}', fontsize=16)
    plt.legend(title='Gerações', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Gerar os gráficos
if not complete_summary.empty:
    # 3.1. Evolução do Fitness Médio ao Longo das Gerações
    fitness_evolution_path = os.path.join(log_directory, 'fitness_evolution_comparative.png')
    plot_fitness_evolution(consolidated_log_df, fitness_evolution_path)
    
    # 3.2. Distribuição da Fitness Máxima na Última Geração
    final_fitness_boxplot_path = os.path.join(log_directory, 'final_fitness_boxplot_comparative.png')
    plot_final_fitness_distribution(consolidated_log_df, final_fitness_boxplot_path)
    
    # 3.3. Comparação da V-measure Média no Treino
    v_measure_train_path = os.path.join(log_directory, 'v_measure_train_comparison.png')
    plot_v_measure_comparison(complete_summary, v_measure_train_path)
    
    # 3.4. Comparação da V-measure Média no Teste
    v_measure_test_path = os.path.join(log_directory, 'v_measure_test_comparison.png')
    plot_v_measure_test_comparison(complete_summary, v_measure_test_path)
    
    # 3.5. Impacto Individual dos Parâmetros
    # Exemplo: Impacto do Population Size na V-measure do Treino
    impact_pop_train_path = os.path.join(log_directory, 'impact_pop_train_v_measure.png')
    plot_parameter_impact(complete_summary, 'Population Size', 'Train_V_Measure_Mean', impact_pop_train_path)
    
    # Exemplo: Impacto das Gerações na V-measure do Treino
    impact_gen_train_path = os.path.join(log_directory, 'impact_gen_train_v_measure.png')
    plot_parameter_impact(complete_summary, 'Generations', 'Train_V_Measure_Mean', impact_gen_train_path)
    
    # Adicione mais análises conforme necessário, por exemplo:
    # Impacto do Population Size na V-measure do Teste
    impact_pop_test_path = os.path.join(log_directory, 'impact_pop_test_v_measure.png')
    plot_parameter_impact(complete_summary, 'Population Size', 'Test_V_Measure_Mean', impact_pop_test_path)
    
    # Impacto das Gerações na V-measure do Teste
    impact_gen_test_path = os.path.join(log_directory, 'impact_gen_test_v_measure.png')
    plot_parameter_impact(complete_summary, 'Generations', 'Test_V_Measure_Mean', impact_gen_test_path)
    
    print("Gráficos gerados e salvos com sucesso na pasta 'logs/'.")
else:
    print("Não há dados consolidados para gerar gráficos.")
