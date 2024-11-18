import random
import operator
import numpy as np
import pandas as pd
from math import sin, cos
from sklearn.metrics import v_measure_score
import os
import csv
import datetime
import time  # Import necessário para medir o tempo

##########################################
# Passo 1: Carregar dados de treinamento #

train_df = pd.read_csv('datasets/breast_cancer_coimbra_train.csv')
train_data = train_df.iloc[:, :-1].values  # -> Atributos
train_labels = train_df.iloc[:, -1].values # -> Classificação

test_df = pd.read_csv('datasets/breast_cancer_coimbra_test.csv')
test_data = test_df.iloc[:, :-1].values  # -> Atributos
test_labels = test_df.iloc[:, -1].values # -> Classificação

# Número de atributos #
num_attributes = train_data.shape[1]

# Número de classes (k) #
k = len(np.unique(train_labels))

# Passo 2: Definição de operadores #
def somar(x, y):
    return x + y

def subtrair(x, y):
    return x - y

def multiplicar(x, y):
    return x * y

def dividir(x, y):
    if y == 0 or x == 0:  # Proteção para evitar divisão por zero
        return 1.0
    else:
        return x / y 

def sen_func(x):
    return sin(x)

def cos_func(x):
    return cos(x)

# Variáveis para exemplo xi
variables_ei = [f'xi{i}' for i in range(num_attributes)]
# Variáveis para exemplo xj
variables_ej = [f'xj{i}' for i in range(num_attributes)]
# Lista completa de variáveis
variables = variables_ei + variables_ej

def generate_random_constant():
    return random.uniform(-1, 1)

# Função para criar a pasta de logs se não existir
if not os.path.exists('logs'):
    os.makedirs('logs')

# Passo 3: Estruturando a árvore
class Node:
    def __init__(self, value, children=[]):
        self.value = value
        self.children = children

    def __str__(self):
        if not self.children:
            return str(self.value)
        else:
            if callable(self.value):
                value_name = self.value.__name__
            else:
                value_name = str(self.value)
            return f"({value_name} " + " ".join(str(child) for child in self.children) + ")"
        
def generate_random_tree(max_depth, depth=0):
    if depth >= max_depth or (depth > 0 and random.random() < 0.1):
        # Criar um terminal
        terminal_choices = variables + [generate_random_constant()]
        terminal_choice = random.choice(terminal_choices)
        return Node(terminal_choice)
    else:
        # Criar um operador
        operator_choices = [somar, subtrair, multiplicar, dividir, sen_func, cos_func]
        operator_choice = random.choice(operator_choices)
        if operator_choice in [sen_func, cos_func]:
            # Operador unário
            child = generate_random_tree(max_depth, depth + 1)
            return Node(operator_choice, [child])
        else:
            # Operador binário
            left_child = generate_random_tree(max_depth, depth + 1)
            right_child = generate_random_tree(max_depth, depth + 1)
            return Node(operator_choice, [left_child, right_child])

# Passo 4: Criar a população inicial
def create_population(size, max_depth):
    population = []
    for i in range(size):
        tree = generate_random_tree(max_depth)
        population.append(tree)
    return population

# Passo 5: Avaliação da árvore
def evaluate_tree(node, xi, xj):
    if isinstance(node.value, str):
        # É uma variável
        if node.value.startswith('xi'):
            index = int(node.value[2:])
            return xi[index]
        elif node.value.startswith('xj'):
            index = int(node.value[2:])
            return xj[index]
        else:
            # É uma constante
            return float(node.value)
    elif isinstance(node.value, (int, float)):
        # É uma constante numérica
        return node.value
    elif callable(node.value):
        # É um operador
        args = [evaluate_tree(child, xi, xj) for child in node.children]
        try:
            result = node.value(*args)
            return result
        except Exception as e:
            #print(f"Erro ao avaliar nó {node}: {e}")
            return 1e6  # Retorna um valor grande em caso de erro
    else:
        raise ValueError(f"Nó inválido: {node.value}")
    

def compute_distance_matrix(tree, data):
    num_samples = len(data)
    distance_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        xi = data[i]
        for j in range(i + 1, num_samples):
            xj = data[j]
            dist = evaluate_tree(tree, xi, xj)
            if np.isnan(dist) or np.isinf(dist):
                dist = 1e6
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Matriz simétrica
    return distance_matrix

def agglomerative_clustering(distance_matrix, num_clusters):
    num_samples = distance_matrix.shape[0]
    # Inicializa cada ponto como um cluster
    clusters = {i: [i] for i in range(num_samples)}
    # Lista de índices dos clusters
    cluster_ids = list(clusters.keys())
    while len(clusters) > num_clusters:
        # Encontrar os dois clusters mais próximos
        min_dist = float('inf')
        cluster_pair = (None, None)
        for i in cluster_ids:
            for j in cluster_ids:
                if i < j:
                    # Calcula a distância média entre os clusters i e j
                    distances = [distance_matrix[p1, p2] for p1 in clusters[i] for p2 in clusters[j]]
                    avg_distance = sum(distances) / len(distances)
                    if avg_distance < min_dist:
                        min_dist = avg_distance
                        cluster_pair = (i, j)
        # Unir os clusters mais próximos
        i, j = cluster_pair
        clusters[i].extend(clusters[j])
        del clusters[j]
        cluster_ids = list(clusters.keys())
    # Atribuir rótulos aos pontos
    labels = np.zeros(num_samples, dtype=int)
    for cluster_label, indices in enumerate(clusters.values()):
        for index in indices:
            labels[index] = cluster_label
    return labels

def evaluate_individual(tree, data, true_labels, num_clusters):
    distance_matrix = compute_distance_matrix(tree, data)
    predicted_labels = agglomerative_clustering(distance_matrix, num_clusters)
    # Calcula a medida V
    v = v_measure_score(true_labels, predicted_labels)
    return v

def copy_tree(node):
    new_node = Node(node.value)
    new_node.children = [copy_tree(child) for child in node.children]
    return new_node

def get_all_nodes(node):
    nodes = [node]
    for child in node.children:
        nodes.extend(get_all_nodes(child))
    return nodes

def tree_depth(node):
    if not node.children:
        return 1
    else:
        return 1 + max(tree_depth(child) for child in node.children)
    
def prune_tree(node, max_depth, current_depth=1):
    if current_depth >= max_depth:
        # Substituir o nó atual por um terminal
        terminal_choices = variables + [generate_random_constant()]
        terminal_choice = random.choice(terminal_choices)
        node.value = terminal_choice
        node.children = []
    else:
        for child in node.children:
            prune_tree(child, max_depth, current_depth + 1)

def crossover(parent1, parent2, max_depth):
    # Copiar as árvores
    child1 = copy_tree(parent1)
    child2 = copy_tree(parent2)
    
    # Obter todos os nós
    nodes1 = get_all_nodes(child1)
    nodes2 = get_all_nodes(child2)
    
    # Selecionar nós aleatórios
    crossover_point1 = random.choice(nodes1)
    crossover_point2 = random.choice(nodes2)
    
    # Trocar subárvores
    crossover_point1.value, crossover_point2.value = crossover_point2.value, crossover_point1.value
    crossover_point1.children, crossover_point2.children = crossover_point2.children, crossover_point1.children
    
    # Verificar profundidade máxima
    if tree_depth(child1) > max_depth:
        prune_tree(child1, max_depth)
    if tree_depth(child2) > max_depth:
        prune_tree(child2, max_depth)
    
    return child1, child2

def mutation(individual, max_depth):
    mutant = copy_tree(individual)
    nodes = get_all_nodes(mutant)
    mutation_point = random.choice(nodes)
    new_subtree = generate_random_tree(max_depth=2)
    mutation_point.value = new_subtree.value
    mutation_point.children = new_subtree.children
    return mutant

def tournament_selection(population, k, tourn_size):
    """
    Realiza a seleção por torneio na população.

    Args:
        population (list): Lista de indivíduos avaliados, cada um é um dicionário com chaves 'tree' e 'fitness'.
        k (int): Número de indivíduos a serem selecionados.
        tourn_size (int): Tamanho do torneio.

    Returns:
        list: Lista de indivíduos selecionados.
    """
    selected = []
    for _ in range(k):
        # Seleciona 'tourn_size' indivíduos aleatoriamente para o torneio
        aspirants = random.sample(population, tourn_size)
        # Encontra o melhor indivíduo entre os aspirantes
        best = max(aspirants, key=lambda ind: ind['fitness'])
        selected.append(best)
    return selected

def genetic_programming(data, labels, k, num_generations, population_size, max_depth, pc, pm, tourn_size, num_elites):
    # Gerar a população inicial
    population = create_population(population_size, max_depth)
    # Avaliar a população
    evaluated_population = []
    for idx, tree in enumerate(population):
        fitness = evaluate_individual(tree, data, labels, k)
        evaluated_population.append({'tree': tree, 'fitness': fitness})
    
    # Inicializar a lista para armazenar os dados de log
    log_data = []
    # Registrar os dados da população inicial
    for idx, individual in enumerate(evaluated_population):
        log_data.append({
            'Generation': 0,
            'Run': current_run,
            'Individual': idx,
            'Fitness': individual['fitness'],
            'Tree': str(individual['tree'])
        })
    
    # Lista para armazenar estatísticas por geração
    stats_data = []
    
    # Elitismo: encontrar os melhores indivíduos
    evaluated_population.sort(key=lambda ind: ind['fitness'], reverse=True)
    elites = evaluated_population[:num_elites]
    best_individual = elites[0]
    print(f"\nMelhor indivíduo inicial: {best_individual['tree']} com fitness {best_individual['fitness']:.4f}")
    
    for generation in range(num_generations):
        generation_start_time = time.time()
        print(f"\n=== Geração {generation + 1} ===")
        
        # Seleção por torneio para o restante da população
        offspring = tournament_selection(evaluated_population, population_size - num_elites, tourn_size)
        print(f"Selecionados {len(offspring)} indivíduos para a reprodução.")
        
        # Aplicar operadores genéticos
        next_generation = []
        i = 0
        while len(next_generation) < (population_size - num_elites) and i < len(offspring):
            rand = random.random()
            if rand < pc and i + 1 < len(offspring):
                # Crossover
                parent1 = offspring[i]['tree']
                parent2 = offspring[i+1]['tree']
                child1_tree, child2_tree = crossover(parent1, parent2, max_depth)
                next_generation.append({'tree': child1_tree, 'fitness': None})
                if len(next_generation) < (population_size - num_elites):
                    next_generation.append({'tree': child2_tree, 'fitness': None})
                i += 2
            elif rand < pc + pm:
                # Mutação
                parent = offspring[i]['tree']
                mutant_tree = mutation(parent, max_depth)
                next_generation.append({'tree': mutant_tree, 'fitness': None})
                i += 1
            else:
                # Reprodução (cópia)
                parent = offspring[i]
                next_generation.append({'tree': copy_tree(parent['tree']), 'fitness': parent['fitness']})
                i += 1
        
        # Garantir que a próxima geração não exceda o tamanho da população
        next_generation = next_generation[:population_size - num_elites]
        
        # Avaliar a nova geração
        for idx, individual in enumerate(next_generation):
            if individual['fitness'] is None:
                individual['fitness'] = evaluate_individual(individual['tree'], data, labels, k)
        
        # Combinar os elitistas com a nova geração
        evaluated_population = elites + next_generation
        
        # Registrar os dados da geração atual
        for idx, individual in enumerate(evaluated_population):
            log_data.append({
                'Generation': generation + 1,
                'Run': current_run,
                'Individual': idx,
                'Fitness': individual['fitness'],
                'Tree': str(individual['tree'])
            })
        
        # Atualizar os elitistas
        evaluated_population.sort(key=lambda ind: ind['fitness'], reverse=True)
        elites = evaluated_population[:num_elites]
        current_best = elites[0]
        if current_best['fitness'] > best_individual['fitness']:
            best_individual = current_best
            print(f"\nNovo melhor indivíduo encontrado: {best_individual['tree']} com fitness {best_individual['fitness']:.4f}")
        
        # Imprimir estatísticas
        fitness_values = [ind['fitness'] for ind in evaluated_population]
        avg_fitness = np.mean(fitness_values)
        max_fitness = fitness_values[0]  # Como a lista está ordenada
        min_fitness = fitness_values[-1]
        generation_time = time.time() - generation_start_time
        print(f"\nEstatísticas da Geração {generation + 1}:")
        print(f"Fitness Médio: {avg_fitness:.4f}, Fitness Máximo: {max_fitness:.4f}, Fitness Mínimo: {min_fitness:.4f}")
        print(f"Tempo da Geração: {generation_time:.2f} segundos")
        
        # Registrar estatísticas da geração
        stats_data.append({
            'Generation': generation + 1,
            'Run': current_run,
            'Average Fitness': avg_fitness,
            'Max Fitness': max_fitness,
            'Min Fitness': min_fitness,
            'Generation Time (s)': generation_time
        })
        
    return best_individual, log_data, stats_data

### **Parâmetros Globais**
# Parâmetros
num_generations = 10 
population_size = 100
max_depth = 7
pc = 0.9  # Probabilidade de crossover
pm = 0.05  # Probabilidade de mutação
tourn_size = 2
num_elites = 5

def evaluate_on_test(individual, data, true_labels, num_clusters):
    print(f"\n=== Avaliando o melhor indivíduo no conjunto de teste ===")
    fitness = evaluate_individual(individual['tree'], data, true_labels, num_clusters)
    print(f"Medida V no conjunto de teste: {fitness:.4f}")
    return fitness

num_runs = 10  # Número de execuções
train_v_measures = []
test_v_measures = []

# Inicializar listas para armazenar os dados de estatísticas de todas as execuções
all_runs_log = []
all_runs_stats = []

# Definir o nome do arquivo de log consolidado baseado nos parâmetros
log_filename = f'logs/log_pop{population_size}_gen{num_generations}.csv'
stats_filename = f'logs/run_summary_pop{population_size}_gen{num_generations}.csv'

# Escrever os cabeçalhos nos arquivos consolidados
with open(log_filename, mode='w', newline='') as csv_file:
    fieldnames = ['Generation', 'Run', 'Individual', 'Fitness', 'Tree']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

with open(stats_filename, mode='w', newline='') as stats_file:
    fieldnames = ['Generation', 'Run', 'Average Fitness', 'Max Fitness', 'Min Fitness', 'Generation Time (s)']
    writer = csv.DictWriter(stats_file, fieldnames=fieldnames)
    writer.writeheader()

for run in range(1, num_runs + 1):
    print(f"\n========== Execução {run} ==========")
    current_run = run  # Variável para identificar a execução atual
    
    start_time = time.time()
    
    best_individual, log_data, stats_data = genetic_programming(
        train_data, train_labels, k, num_generations, population_size,
        max_depth, pc, pm, tourn_size, num_elites
    )
    
    v_train = best_individual['fitness']
    v_test = evaluate_on_test(best_individual, test_data, test_labels, k)
    train_v_measures.append(v_train)
    test_v_measures.append(v_test)
    print(f"Execução {run}: V-train = {v_train:.4f}, V-test = {v_test:.4f}")
    
    # Salvar os dados de log no arquivo consolidado
    with open(log_filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Generation', 'Run', 'Individual', 'Fitness', 'Tree'])
        writer.writerows(log_data)
    
    # Salvar os dados de estatísticas no arquivo consolidado
    with open(stats_filename, mode='a', newline='') as stats_file:
        writer = csv.DictWriter(stats_file, fieldnames=['Generation', 'Run', 'Average Fitness', 'Max Fitness', 'Min Fitness', 'Generation Time (s)'])
        writer.writerows(stats_data)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tempo total da Execução {run}: {execution_time:.2f} segundos")

# Calcular as médias e desvios padrão das medidas V
mean_train_v = np.mean(train_v_measures)
std_train_v = np.std(train_v_measures)
mean_test_v = np.mean(test_v_measures)
std_test_v = np.std(test_v_measures)

print("\n=== Resultados Finais ===")
print(f"Média V-measure Treino: {mean_train_v:.4f} ± {std_train_v:.4f}")
print(f"Média V-measure Teste: {mean_test_v:.4f} ± {std_test_v:.4f}")

# Salvar os resultados finais em um arquivo de resumo
summary_filename = f'logs/summary_pop{population_size}_gen{num_generations}.csv'
with open(summary_filename, mode='w', newline='') as summary_file:
    writer = csv.writer(summary_file)
    writer.writerow(['Run', 'Train V-measure', 'Test V-measure'])
    for run in range(1, num_runs + 1):
        writer.writerow([run, train_v_measures[run - 1], test_v_measures[run - 1]])
    
    writer.writerow([])
    writer.writerow([
        'Average Train V-measure', 'Std Dev Train V-measure',
        'Average Test V-measure', 'Std Dev Test V-measure'
    ])
    writer.writerow([mean_train_v, std_train_v, mean_test_v, std_test_v])
