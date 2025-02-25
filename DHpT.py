import random
from deap import base, creator, tools
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
import matplotlib.pyplot as plt

# from case_size10_net1 import get_importances, get_scores, TS_data, time_points, gene_names, regulators, SS_data, \
#     gold_edges


def get_links(VIM, gene_names, regulators, sort=True, file_name=None):
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

    # Create a list of predicted edges
    pred_edges = [(gene_names[j], gene_names[i], score)
                  for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]

    # Convert to DataFrame
    pred_edges = pd.DataFrame(pred_edges, columns=['Gene1', 'Gene2', 'Importance'])

    # Sort the DataFrame by Importance if required
    if sort:
        pred_edges.sort_values(by='Importance', ascending=False, inplace=True)

    # save to file, otherwise return the DataFrame
    if file_name is None:
        return pred_edges
    else:
        pred_edges.to_csv(file_name, sep='\t', header=None, index=None)
        return None  # Avoid returning a DataFrame if saving to a file


def get_importances(TS_data, time_points, time_lag, gene_names, regulators, alpha, SS_data=None, param={}):
    time_start = time.time()

    ngenes = TS_data[0].shape[1]

    alphas = [alpha] * ngenes

    # Get the indices of the candidate regulators
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
    VIM = np.zeros((ngenes, ngenes))

    for i in range(ngenes):
        input_idx = idx.copy()
        if i in input_idx:
            input_idx.remove(i)
        vi = get_importances_single(TS_data, time_points, time_lag, alphas[i], input_idx, i, SS_data, param)
        VIM[i, :] = vi

    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM


def get_importances_single(TS_data, time_points, time_lag, alpha, input_idx, output_idx, SS_data, param):
    h = 1  # lag (in number of time points) used for the finite approximation of the derivative of the target gene expression
    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)
    nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data])
    ninputs = len(input_idx)
    # Construct learning sample
    # Time-series data
    input_matrix_time = np.zeros((nsamples_time - h * nexp, ninputs))
    output_vect_time = np.zeros(nsamples_time - h * nexp)
    nsamples_count = 0
    for (i, current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]
        npoints = current_timeseries.shape[0]
        time_diff_current = current_time_points[h:] - current_time_points[:npoints - h]
        current_timeseries_input = current_timeseries[:npoints - h, input_idx]
        current_timeseries_output = (current_timeseries[h:, output_idx] - current_timeseries[:npoints - h,
                                                                          output_idx]) / time_diff_current + alpha * current_timeseries[
                                                                                                                     :npoints - h,
                                                                                                                     output_idx]

        # Time delay
        npoints = current_timeseries_input.shape[0]
        current_timeseries_input = current_timeseries_input[:npoints - time_lag]
        current_timeseries_output = current_timeseries_output[time_lag:]

        nsamples_current = current_timeseries_input.shape[0]
        input_matrix_time[nsamples_count:nsamples_count + nsamples_current, :] = current_timeseries_input
        output_vect_time[nsamples_count:nsamples_count + nsamples_current] = current_timeseries_output
        nsamples_count += nsamples_current

    # Steady-state data
    if SS_data is not None:
        input_matrix_steady = SS_data[:, input_idx]
        output_vect_steady = SS_data[:, output_idx] * alpha

        # Concatenation
        input_all = np.vstack([input_matrix_steady, input_matrix_time])
        output_all = np.concatenate((output_vect_steady, output_vect_time))
    else:
        input_all = input_matrix_time
        output_all = output_vect_time

    treeEstimator = XGBRegressor(**param)

    # Learn ensemble of trees
    treeEstimator.fit(input_all, output_all)

    # Compute importance scores
    feature_importances = treeEstimator.feature_importances_
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances

    return vi


def get_scores(VIM, gold_edges, gene_names, regulators):
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    # pred_edges = pred_edges.iloc[:100000]
    final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
    auroc = roc_auc_score(final['2_y'], final['2_x'])
    aupr = average_precision_score(final['2_y'], final['2_x'])
    return auroc, aupr


TS_data = pd.read_csv(
    "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_timeseries.tsv",
    sep='\t').values
SS_data_1 = pd.read_csv(
    "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_knockouts.tsv",
    sep='\t').values
SS_data_2 = pd.read_csv(
    "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_knockdowns.tsv",
    sep='\t').values

# SS_data_3 = pd.read_csv("/Users/macbookpro/Documents/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_multifactorial.tsv",sep='\t').values


# get the steady-state data
SS_data = np.vstack([SS_data_1, SS_data_2])

# get the steady-state data
SS_data = np.vstack([SS_data_1])

i = np.arange(0, 85, 21)
j = np.arange(21, 106, 21)

# get the time-series data
TS_data = [TS_data[i:j] for (i, j) in zip(i, j)]
# get time points
time_points = [np.arange(0, 1001, 50)] * 5

ngenes = TS_data[0].shape[1]
gene_names = ['G' + str(i + 1) for i in range(ngenes)]
regulators = gene_names.copy()

gold_edges = pd.read_csv(
    "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_goldstandard.tsv",
    '\t', header=None)

xgb_kwargs = dict(n_estimators=398, learning_rate=0.0137089260215423, importance_type="weight", max_depth=5, n_j123obs=-1,
                  objective='reg:squarederror')

VIM = get_importances(TS_data, time_points, time_lag=0, gene_names=gene_names, regulators=regulators,
                      alpha=0.022408670532763, SS_data=SS_data, param=xgb_kwargs)
auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)
# print(auroc, aupr)

# get_links to get the predicted network
predicted_edges = get_links(VIM, gene_names, regulators, sort=True, file_name=None)

# Helper functions (your existing functions remain the same)
# def get_importances(TS_data, time_points, time_lag, gene_names, regulators, alpha, SS_data=None, param={}):
#     time_start = time.time()
#     ngenes = TS_data[0].shape[1]
#     alphas = [alpha] * ngenes
#
#     idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
#     VIM = np.zeros((ngenes, ngenes))
#
#     for i in range(ngenes):
#         input_idx = idx.copy()
#         if i in input_idx:
#             input_idx.remove(i)
#         vi = get_importances_single(TS_data, time_points, time_lag, alphas[i], input_idx, i, SS_data, param)
#         VIM[i, :] = vi
#
#     time_end = time.time()
#     print("Elapsed time: %.2f seconds" % (time_end - time_start))
#
#     return VIM

def get_importances_single(TS_data, time_points, time_lag, alpha, input_idx, output_idx, SS_data, param):
    h = 1
    ngenes = TS_data[0].shape[1]
    nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data])
    ninputs = len(input_idx)

    input_matrix_time = np.zeros((nsamples_time - h * len(TS_data), ninputs))
    output_vect_time = np.zeros(nsamples_time - h * len(TS_data))
    nsamples_count = 0

    for i, current_timeseries in enumerate(TS_data):
        current_time_points = time_points[i]
        npoints = current_timeseries.shape[0]
        time_diff_current = current_time_points[h:] - current_time_points[:npoints - h]
        current_timeseries_input = current_timeseries[:npoints - h, input_idx]
        current_timeseries_output = (current_timeseries[h:, output_idx] - current_timeseries[:npoints - h,
                                                                          output_idx]) / time_diff_current + alpha * current_timeseries[
                                                                                                                     :npoints - h,
                                                                                                                     output_idx]

        npoints = current_timeseries_input.shape[0]
        current_timeseries_input = current_timeseries_input[:npoints - time_lag]
        current_timeseries_output = current_timeseries_output[time_lag:]

        nsamples_current = current_timeseries_input.shape[0]
        input_matrix_time[nsamples_count:nsamples_count + nsamples_current, :] = current_timeseries_input
        output_vect_time[nsamples_count:nsamples_count + nsamples_current] = current_timeseries_output
        nsamples_count += nsamples_current

    if SS_data is not None:
        input_matrix_steady = SS_data[:, input_idx]
        output_vect_steady = SS_data[:, output_idx] * alpha
        input_all = np.vstack([input_matrix_steady, input_matrix_time])
        output_all = np.concatenate((output_vect_steady, output_vect_time))
    else:
        input_all = input_matrix_time
        output_all = output_vect_time

    treeEstimator = XGBRegressor(**param)
    treeEstimator.fit(input_all, output_all)

    feature_importances = treeEstimator.feature_importances_
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances

    return vi

# def get_scores(VIM, gold_edges, gene_names, regulators):
#     idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
#     pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
#     pred_edges = pd.DataFrame(pred_edges)
#     final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
#     auroc = roc_auc_score(final['2_y'], final['2_x'])
#     aupr = average_precision_score(final['2_y'], final['2_x'])
#     return auroc, aupr

def evaluate(individual, TS_data, time_points, time_lag, gene_names, regulators, SS_data, gold_edges):
    """
    Evaluate the fitness of an individual based on the AUC score and AP score.
    The individual represents a set of hyperparameters for XGBoost.
    """
    # Unpack individual hyperparameters
    n_estimators, learning_rate, max_depth = individual

    # Define the XGBoost parameters
    xgb_kwargs = {
        'n_estimators': int(n_estimators),
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'importance_type': "weight",
        'n_jobs': -1,
        'objective': 'reg:squarederror'
    }

    # Get the importance values using the current hyperparameters
    VIM = get_importances(TS_data, time_points, time_lag, gene_names, regulators, alpha=0.022408670532763,
                          SS_data=SS_data, param=xgb_kwargs)

    # Calculate AUROC and AUPR scores
    auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)

    # Return a tuple as expected by DEAP (fitness values)
    return auroc, aupr


# Genetic algorithm setup with elitism
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))  # Maximize both AUROC and AUPR
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Define the bounds for the hyperparameters
BOUNDS_LOW = [50, 0.001, 3]  # n_estimators: [50, 500], learning_rate: [0.001, 0.1], max_depth: [3, 15]
BOUNDS_HIGH = [500, 0.1, 15]

# Register the creation of individuals with random values within bounds
for i in range(3):  # We have 3 hyperparameters
    toolbox.register(f"hyperparameter_{i}", random.uniform, BOUNDS_LOW[i], BOUNDS_HIGH[i])

# Create individuals based on the hyperparameter bounds
toolbox.register("individualCreator", tools.initCycle, creator.Individual,
                 tuple([toolbox.__getattribute__(f"hyperparameter_{i}") for i in range(3)]), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individualCreator)

# Register crossover, mutation, and selection methods
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=1.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=1.0, indpb=1.0/3)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("evaluate", evaluate, TS_data=TS_data, time_points=time_points, time_lag=0, gene_names=gene_names,
                 regulators=regulators, SS_data=SS_data, gold_edges=gold_edges)

# Set GA parameters
population_size = 30
generations = 10
cx_probability = 0.7
mut_probability = 0.3

# Create an initial population
population = toolbox.population(n=population_size)

# Hall of Fame for elitism
hof = tools.HallOfFame(1)

# Define the statistics (optional)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Run the genetic algorithm with elitism using eaSimple
for gen in range(generations):
    print(f"Generation {gen}")

    # Evaluate all individuals in the population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Select the best individuals for mating
    selected = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, selected))

    # Apply crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx_probability:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mut_probability:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the fitness of the new offspring
    for ind in offspring:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    # Update the Hall of Fame
    hof.update(offspring)

    # Replace the old population with the new one
    population[:] = offspring

    # Log statistics
    record = stats.compile(population)
    print(record)

# After running GA, output the best individual
best_individual = hof[0]
print(f"Best individual: {best_individual}")
print(f"Fitness: {best_individual.fitness.values}")

#Calculate partial correlation
def calculate_partial_correlations(data_matrix):
    """
    Calculate the partial correlation matrix.

    :param data_matrix: 2D numpy array where rows are samples and columns are variables
    :return: Partial correlation matrix
    """
    # Calculate the inverse of the covariance matrix
    covariance_matrix = np.cov(data_matrix, rowvar=False)
    precision_matrix = np.linalg.inv(covariance_matrix)

    # Compute the partial correlation matrix
    d = np.sqrt(np.diag(precision_matrix))
    partial_corr_matrix = -precision_matrix / np.outer(d, d)

    # Set the diagonal elements to 1
    np.fill_diagonal(partial_corr_matrix, 1)

    return partial_corr_matrix


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial.distance import pdist, squareform

def adjust_node_positions(pos, min_distance=0.2, max_iterations=100):
    """
    Adjusts node positions to avoid overlaps using a simple repulsion mechanism.
    
    :param pos: Dictionary of node positions
    :param min_distance: Minimum allowable distance between nodes
    :param max_iterations: Maximum number of adjustments
    :return: Adjusted node positions
    """
    nodes = list(pos.keys())
    positions = np.array([pos[n] for n in nodes])
    
    for _ in range(max_iterations):
        distances = squareform(pdist(positions))  # Compute pairwise distances
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        
        overlap_indices = np.where(distances < min_distance)
        if len(overlap_indices[0]) == 0:
            break  # Exit early if no overlaps
        
        for i, j in zip(*overlap_indices):
            if i < j:  # Avoid double adjustments
                vec = positions[j] - positions[i]
                norm = np.linalg.norm(vec)
                if norm == 0:
                    vec = np.random.rand(2) - 0.5  # Random small shift if nodes are exactly on top
                    norm = np.linalg.norm(vec)
                displacement = (min_distance - norm) * (vec / norm) * 0.5
                positions[i] -= displacement
                positions[j] += displacement

    return {nodes[i]: tuple(positions[i]) for i in range(len(nodes))}

def plot_network_with_autocorrelation(correlation_matrix, gene_names, threshold=0.3):
    """
    Visualizes the gene regulatory network (GRN) with non-overlapping nodes and edges.
    """
    G = nx.DiGraph()
    
    # Add nodes
    G.add_nodes_from(gene_names)
    
    # Add edges based on correlation matrix
    num_genes = len(gene_names)
    for i in range(num_genes):
        for j in range(num_genes):
            if i != j:  
                corr = correlation_matrix[i, j]
                if abs(corr) >= threshold:
                    G.add_edge(gene_names[i], gene_names[j], weight=corr)

    G.remove_nodes_from(list(nx.isolates(G)))  # Remove nodes with no edges
    
    if len(G.nodes) == 0:
        print("No significant connections found above the threshold.")
        return
    
    # Compute initial positions with high repulsion
    pos = nx.spring_layout(G, k=0.4, iterations=50)  # Larger k increases spacing
    pos = adjust_node_positions(pos, min_distance=0.2)  # Ensure no overlaps

    # Draw nodes
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    ax = plt.gca()
    node_radius = 0.05  
    base_width = 1.5  
    scale_factor = 2.5  

    edge_offsets = {}  

    for u, v, d in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        weight = d['weight']
        
        direction = np.array([x2 - x1, y2 - y1])
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue 
        
        unit_direction = direction / norm
        start = np.array([x1, y1]) + unit_direction * node_radius
        end = np.array([x2, y2]) - unit_direction * node_radius

        edge_key = tuple(sorted([u, v]))  
        if edge_key in edge_offsets:
            offset = edge_offsets[edge_key]
        else:
            offset = np.random.uniform(-0.03, 0.03)  
            edge_offsets[edge_key] = offset

        perp_direction = np.array([-unit_direction[1], unit_direction[0]]) * offset
        start += perp_direction
        end += perp_direction

        linewidth = base_width + scale_factor * abs(weight)
        
        if weight > 0:
            arrowstyle = "-|>"  
            color = "black"
        else:
            arrowstyle = "-["  
            color = "red"

        arrow = mpatches.FancyArrowPatch(
            start, end,
            arrowstyle=arrowstyle,
            mutation_scale=15,
            color=color,
            lw=linewidth,
            connectionstyle="arc3,rad=0.2"  
        )
        ax.add_patch(arrow)

    plt.title(f"Gene Regulatory Network (Threshold ≥ {threshold})")
    plt.show()


# After the GA loop and identifying the best individual:
best_individual = hof[0]
# print(f"Best individual: {best_individual}")
# print(f"Fitness: {best_individual.fitness.values}")

# Get the predicted network using the best hyperparameters
xgb_kwargs_best = xgb_kwargs

# Get the importance values using the best hyperparameters
VIM_best = get_importances(TS_data, time_points, time_lag=0, gene_names=gene_names, regulators=regulators,
                           alpha=0.022408670532763, SS_data=SS_data, param=xgb_kwargs_best)

# Get predicted edges (GRN) from the best individual
predicted_edges_best = get_links(VIM_best, gene_names, regulators, sort=True, file_name=None)

# Plot the network for the best individual
plot_network_with_autocorrelation(predicted_edges_best, gene_names)
