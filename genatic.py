import random
import numpy as np
from deap import base, creator, tools
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from case_size10_net1 import get_importances, TS_data, time_points, gene_names, SS_data, regulators, get_scores, \
    gold_edges

# Define the boundaries for hyperparameters of XGBRegressor
BOUNDS_LOW = [1, 0.01, 1]  # lower bounds: n_estimators, learning_rate, max_depth
BOUNDS_HIGH = [1000, 1.0, 15]  # upper bounds: n_estimators, learning_rate, max_depth

NUM_OF_PARAMS = len(BOUNDS_HIGH)  # number of hyperparameters to optimize

# GA constants
POPULATION_SIZE = 20
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.2  # probability for mutating an individual
MAX_GENERATIONS = 5
HALL_OF_FAME_SIZE = 5
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Define the fitness strategy for maximizing the AUROC score
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create the toolbox and register necessary functions
toolbox = base.Toolbox()

# Define the hyperparameter attributes (n_estimators, learning_rate, max_depth)
for i in range(NUM_OF_PARAMS):
    toolbox.register("hyperparameter_" + str(i),
                     random.uniform, BOUNDS_LOW[i], BOUNDS_HIGH[i])

# Create the tuple containing the attribute generators for each parameter
hyperparameters = ()
for i in range(NUM_OF_PARAMS):
    hyperparameters = hyperparameters + (toolbox.__getattribute__("hyperparameter_" + str(i)),)

# Register individual and population creation methods
toolbox.register("individualCreator", tools.initCycle, creator.Individual, hyperparameters, n=1)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# Define the fitness evaluation function (AUROC and AUPR score)
def evaluate(individual):
    # Extract hyperparameters
    param = {
        'n_estimators': int(individual[0]),  # n_estimators
        'learning_rate': individual[1],  # learning_rate
        'max_depth': int(individual[2]),  # max_depth
        'objective': 'reg:squarederror',
        'importance_type': 'weight',
        'n_jobs': -1
    }

    # `get_importances` and `get_scores` functions are already defined
    VIM = get_importances(TS_data, time_points, time_lag=0, gene_names=gene_names, regulators=regulators,
                          alpha=0.022408670532763, SS_data=SS_data, param=param)

    # Calculate AUROC and AUPR scores
    auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)

    # We return a tuple, as DEAP expects it
    return auroc,  # Maximizing AUROC


# Register the evaluation function
toolbox.register("evaluate", evaluate)

# Genetic operators: Selection, Crossover, and Mutation
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=CROWDING_FACTOR,
                 indpb=1.0 / NUM_OF_PARAMS)


# Create a function to run the GA
def run_genetic_algorithm():
    # Create the initial population
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # Prepare the statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # Hall of Fame to store the best solutions
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # Run the GA with elitism and genetic operations
    population, logbook = tools.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                         ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # Return the best individual from the Hall of Fame
    best_individual = hof.items[0]
    return best_individual, logbook


# Main function to execute the GA and print results
if __name__ == "__main__":
    # Run the Genetic Algorithm
    best_params, logbook = run_genetic_algorithm()

    # Print the best hyperparameters and their AUROC score
    print(f"Best Hyperparameters: {best_params}")
    print(f"AUROC Score: {best_params.fitness.values[0]}")

    # Plot the fitness statistics over generations
    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")

    sns.set_style("whitegrid")
    plt.plot(max_fitness_values, color='red', label="Max Fitness")
    plt.plot(mean_fitness_values, color='green', label="Average Fitness")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.legend()

    # Ensure the plot is shown
    plt.tight_layout()
    plt.show()
