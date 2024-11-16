import random
import operator
import numpy as np
import pandas as pd
from math import sin, cos
from collections import deque

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
    if y == 0 or x == 0: # -> Proteção pra não bugar em casos de zero (apesar de bem improvável)
        return 1.0
    else:
        return x / y 
    
def sen_func(x):
    return sin(x)

def cos_func(x):
    return cos(x)

# Variáveis para exemplo ei
variables_ei = [f'xi{i}' for i in range(num_attributes)]
# Variáveis para exemplo ej
variables_ej = [f'xj{i}' for i in range(num_attributes)]
# Lista completa de variáveis
variables = variables_ei + variables_ej

def generate_random_constant():
    return random.uniform(-1, 1)

# Estruturando a árvore
#      o
#      |
#     / \
#    o   o

