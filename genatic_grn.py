import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBRegressor
import networkx as nx
from matplotlib import pyplot as plt
import time

from case_size10_net1 import get_importances, TS_data, time_points, gene_names, regulators, SS_data, get_scores, \
    gold_edges, get_links

# DEAP Setup
# Create fitness function and individual structure
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define hyperparameter bounds for tuning
BOUNDS_LOW = [1, 0.01, 3]  # n_estimators, learning_rate, max_depth
BOUNDS_HIGH = [1000, 0.5, 10]  # n_estimators, learning_rate, max_depth

NUM_OF_PARAMS = len(BOUNDS_LOW)


# Create individual (hyperparameter set)
def create_individual():
    """Generate a random individual (a list of hyperparameters)"""
    return [random.uniform(BOUNDS_LOW[i], BOUNDS_HIGH[i]) for i in range(NUM_OF_PARAMS)]


# Fitness function to evaluate the individual
def evaluate(individual):
    """Evaluate the fitness of the individual (combination of hyperparameters)"""
    # Extract hyperparameters from individual
    n_estimators = int(individual[0])
    learning_rate = individual[1]
    max_depth = int(individual[2])

    # Set up XGBoost parameters
    xgb_kwargs = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'importance_type': 'weight',
        'n_jobs': -1,
        'objective': 'reg:squarederror'
    }

    # Compute the Gene Regulatory Network importance matrix
    VIM = get_importances(TS_data, time_points, time_lag=0, gene_names=gene_names, regulators=regulators,
                          alpha=0.022408670532763, SS_data=SS_data, param=xgb_kwargs)

    # Get AUROC and AUPR scores
    auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)

    # Print the AUROC and AUPR for the current individual
    print(f"AUROC: {auroc:.4f}, AUPR: {aupr:.4f} for individual: {individual}")

    # Return a tuple since DEAP expects it
    return auroc,  # Maximizing AUROC


# Mutation function (randomly modify a hyperparameter)
def mutate(individual):
    """Random mutation on the individual (hyperparameters)"""
    param_idx = random.randint(0, NUM_OF_PARAMS - 1)
    # Apply mutation within bounds
    individual[param_idx] = random.uniform(BOUNDS_LOW[param_idx], BOUNDS_HIGH[param_idx])
    return individual,


# Crossover function (swap parts of two individuals)
def crossover(ind1, ind2):
    """Crossover between two individuals"""
    # Pick a random point for crossover
    cxpoint = random.randint(1, NUM_OF_PARAMS - 1)

    # Swap the genes after the crossover point
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return ind1, ind2


# Set up DEAP Toolbox
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# Parameters for the GA
POPULATION_SIZE = 10  # Population size
P_CROSSOVER = 0.9  # Probability of crossover
P_MUTATION = 0.2  # Probability of mutation
MAX_GENERATIONS = 5  # Number of generations
HALL_OF_FAME_SIZE = 1  # Track the best individual

# Create the initial population
population = toolbox.population(n=POPULATION_SIZE)

# Create the Hall of Fame to store the best individual
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# Define statistics to track during the evolution
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

# Run the Genetic Algorithm
start_time = time.time()

population, logbook = algorithms.eaSimple(
    population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
    stats=stats, halloffame=hof, verbose=True)

end_time = time.time()
print(f"Total time for GA: {end_time - start_time:.2f} seconds")

# Get the best hyperparameters
best_individual = hof[0]
print(f"\nBest individual (hyperparameters): {best_individual}")
print(f"Best AUROC: {best_individual.fitness.values[0]}")

# Visualize the network using the best individual hyperparameters
n_estimators = int(best_individual[0])
learning_rate = best_individual[1]
max_depth = int(best_individual[2])

# Set up XGBoost parameters
xgb_kwargs = {
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'max_depth': max_depth,
    'importance_type': 'weight',
    'n_jobs': -1,
    'objective': 'reg:squarederror'
}

# Compute the Gene Regulatory Network importance matrix using best hyperparameters
VIM = get_importances(TS_data, time_points, time_lag=0, gene_names=gene_names, regulators=regulators,
                      alpha=0.022408670532763, SS_data=SS_data, param=xgb_kwargs)

# Get the predicted network links
predicted_edges = get_links(VIM, gene_names, regulators, sort=True, file_name=None)

# Check if predicted_edges is not None
if predicted_edges is not None:
    # Create a network from the predicted edges using NetworkX
    G = nx.DiGraph()  # Directed graph for regulatory interactions

    # Adding edges from the predicted edges DataFrame to the graph
    for index, row in predicted_edges.iterrows():
        G.add_edge(row['Gene1'], row['Gene2'], weight=row['Importance'])

    # Visualize the network
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)  # layout of the network
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=10, font_weight='bold',
            arrows=True)
    plt.title("Gene Regulatory Network (GRN) Predicted by XGBoost")
    plt.show()
else:
    print("No predicted edges found.")
