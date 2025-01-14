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

# Function to get predicted links (edges) based on feature importance
def get_links(VIM, gene_names, regulators, sort=True, file_name=None):
    # Identify indices of the regulator genes
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

    # Create a list of predicted edges with their importance scores
    pred_edges = [(gene_names[j], gene_names[i], score)
                  for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]

    # Convert the edges to a DataFrame
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


# Function to compute the importance values for each gene based on XGBoost
def get_importances(TS_data, time_points, time_lag, gene_names, regulators, alpha, SS_data=None, param={}):
    time_start = time.time()

    # Initialize variables
    ngenes = TS_data[0].shape[1] # Number of genes
    alphas = [alpha] * ngenes # List of alpha values (one per gene)

    # Get the indices of the candidate regulators
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

    # Initialize a matrix to store importance values
    VIM = np.zeros((ngenes, ngenes))

    # Iterate through each gene and compute its feature importances
    for i in range(ngenes):
        input_idx = idx.copy()
        if i in input_idx:
            input_idx.remove(i)
        vi = get_importances_single(TS_data, time_points, time_lag, alphas[i], input_idx, i, SS_data, param)
        VIM[i, :] = vi

    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM

# Function to compute importance for a single gene using XGBoost
def get_importances_single(TS_data, time_points, time_lag, alpha, input_idx, output_idx, SS_data, param):

    # Prepare data matrices for XGBoost model training
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

    # Construct the time-series data matrix
    for (i, current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]
        npoints = current_timeseries.shape[0]
        time_diff_current = current_time_points[h:] - current_time_points[:npoints - h]
        current_timeseries_input = current_timeseries[:npoints - h, input_idx]
        current_timeseries_output = (current_timeseries[h:, output_idx] - current_timeseries[:npoints - h,
                                                                          output_idx]) / time_diff_current + alpha * current_timeseries[
                                                                                                                     :npoints - h,
                                                                                                                     output_idx]

        # Apply time lag
        npoints = current_timeseries_input.shape[0]
        current_timeseries_input = current_timeseries_input[:npoints - time_lag]
        current_timeseries_output = current_timeseries_output[time_lag:]

        nsamples_current = current_timeseries_input.shape[0]
        input_matrix_time[nsamples_count:nsamples_count + nsamples_current, :] = current_timeseries_input
        output_vect_time[nsamples_count:nsamples_count + nsamples_current] = current_timeseries_output
        nsamples_count += nsamples_current

    # If steady-state data is provided, include it as well
    if SS_data is not None:
        input_matrix_steady = SS_data[:, input_idx]
        output_vect_steady = SS_data[:, output_idx] * alpha

        # Concatenation
        input_all = np.vstack([input_matrix_steady, input_matrix_time])
        output_all = np.concatenate((output_vect_steady, output_vect_time))
    else:
        input_all = input_matrix_time
        output_all = output_vect_time

    # Train an XGBoost model to compute feature importances
    treeEstimator = XGBRegressor(**param)

    # Learn ensemble of trees
    treeEstimator.fit(input_all, output_all)

    # Compute importance scores
    feature_importances = treeEstimator.feature_importances_
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances

    return vi

# Function to compute AUROC and AUPR scores
def get_scores(VIM, gold_edges, gene_names, regulators):

    # Extract the predicted edges based on the feature importance matrix

    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    # pred_edges = pred_edges.iloc[:100000]

    # Merge predicted edges with gold edges
    final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')

    # Calculate AUROC and AUPR scores
    auroc = roc_auc_score(final['2_y'], final['2_x'])
    aupr = average_precision_score(final['2_y'], final['2_x'])
    return auroc, aupr

# Load the time-series and steady-state data
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

# Organize time-series data into intervals
i = np.arange(0, 85, 21)
j = np.arange(21, 106, 21)

# get the time-series data
TS_data = [TS_data[i:j] for (i, j) in zip(i, j)]
# get time points
time_points = [np.arange(0, 1001, 50)] * 5

# Define gene names and regulators (all genes in this case)
ngenes = TS_data[0].shape[1]
gene_names = ['G' + str(i + 1) for i in range(ngenes)]
regulators = gene_names.copy()

# Compute AUROC and AUPR scores using gold edges (use a placeholder here)
gold_edges = pd.read_csv(
    "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_goldstandard.tsv",
    '\t', header=None)


# Compute feature importance matrix
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

# def get_importances_single(TS_data, time_points, time_lag, alpha, input_idx, output_idx, SS_data, param):
#     h = 1
#     ngenes = TS_data[0].shape[1]
#     nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data])
#     ninputs = len(input_idx)
#
#     input_matrix_time = np.zeros((nsamples_time - h * len(TS_data), ninputs))
#     output_vect_time = np.zeros(nsamples_time - h * len(TS_data))
#     nsamples_count = 0
#
#     for i, current_timeseries in enumerate(TS_data):
#         current_time_points = time_points[i]
#         npoints = current_timeseries.shape[0]
#         time_diff_current = current_time_points[h:] - current_time_points[:npoints - h]
#         current_timeseries_input = current_timeseries[:npoints - h, input_idx]
#         current_timeseries_output = (current_timeseries[h:, output_idx] - current_timeseries[:npoints - h,
#                                                                           output_idx]) / time_diff_current + alpha * current_timeseries[
#                                                                                                                      :npoints - h,
#                                                                                                                      output_idx]
#
#         npoints = current_timeseries_input.shape[0]
#         current_timeseries_input = current_timeseries_input[:npoints - time_lag]
#         current_timeseries_output = current_timeseries_output[time_lag:]
#
#         nsamples_current = current_timeseries_input.shape[0]
#         input_matrix_time[nsamples_count:nsamples_count + nsamples_current, :] = current_timeseries_input
#         output_vect_time[nsamples_count:nsamples_count + nsamples_current] = current_timeseries_output
#         nsamples_count += nsamples_current
#
#     if SS_data is not None:
#         input_matrix_steady = SS_data[:, input_idx]
#         output_vect_steady = SS_data[:, output_idx] * alpha
#         input_all = np.vstack([input_matrix_steady, input_matrix_time])
#         output_all = np.concatenate((output_vect_steady, output_vect_time))
#     else:
#         input_all = input_matrix_time
#         output_all = output_vect_time
#
#     treeEstimator = XGBRegressor(**param)
#     treeEstimator.fit(input_all, output_all)
#
#     feature_importances = treeEstimator.feature_importances_
#     vi = np.zeros(ngenes)
#     vi[input_idx] = feature_importances
#
#     return vi

# def get_scores(VIM, gold_edges, gene_names, regulators):
#     idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
#     pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
#     pred_edges = pd.DataFrame(pred_edges)
#     final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
#     auroc = roc_auc_score(final['2_y'], final['2_x'])
#     aupr = average_precision_score(final['2_y'], final['2_x'])
#     return auroc, aupr

#Evaluate fitness
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
generations = 2
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

# Print the best individual's hyperparameters
print(f"Best individual: {best_individual}")
print(f"Hyperparameter values:")
print(f"  n_estimators: {int(best_individual[0])}")
print(f"  learning_rate: {best_individual[1]:.6f}")
print(f"  max_depth: {int(best_individual[2])}")
print(f"Fitness: {best_individual.fitness.values}")


# Print the fitness values (AUROC, AUPR)
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

#draw network showing attribution and inhibution
def draw_network_with_attribution_inhibition(correlation_matrix, gene_names, threshold=0.25):
    """
    Draw a network graph based on the partial correlation matrix, showing attribution (+) and inhibition (-).

    :param correlation_matrix: 2D numpy array of partial correlations
    :param gene_names: List of gene names corresponding to the matrix
    :param threshold: Minimum correlation value to include an edge
    """
    # Create a directed graph
    G = nx.DiGraph()  # Use directed graph for arrows
    filtered_edges = predicted_edges[predicted_edges['Importance'] >= threshold]

    # Add nodes to the graph (even for genes with no correlation)
    G.add_nodes_from(gene_names)

    # Add edges based on partial correlation threshold
    for i in range(len(filtered_edges)):
        gene1 = filtered_edges.iloc[i]['Gene1']
        gene2 = filtered_edges.iloc[i]['Gene2']
        importance = filtered_edges.iloc[i]['Importance']

        corr = correlation_matrix[gene_names.index(gene1), gene_names.index(gene2)]
        sign = "-|-" if corr > 0 else "----|"  # Attribution (+) or Inhibition (-)

        # Add edge with weight and sign
        if corr > 0:
            G.add_edge(gene1, gene2, weight=abs(corr), sign=sign)  # Positive correlation -> gene1 -> gene2
        elif corr < 0:
            G.add_edge(gene2, gene1, weight=abs(corr), sign=sign)  # Negative correlation -> gene2 -> gene1

    # Create edge color scheme based on attribution and inhibition
    edge_colors = ['green' if G[u][v]['sign'] == "-|-" else 'red' for u, v in
                   G.edges()]  # Green for attribution (+), red for inhibition (-)
    edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]  # Scale edge widths based on correlation strength

    # Plot the network with arrows
    pos = nx.spring_layout(G)  # Positioning for nodes
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color='lightblue',
        edge_color=edge_colors,
        width=edge_widths,
        font_size=10,
        arrows=True,  # Enable arrows on directed edges
        node_size=2000,
        font_weight='bold',
    )

    # Label edges with attribution (+) and inhibition (-)
    edge_labels = {
        (u, v): f"{G[u][v]['sign']}" for u, v in G.edges()
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"Gene Network (Threshold: {threshold})")
    plt.show()


# Convert TS_data into a single 2D array for correlation calculation
expression_data = np.vstack(TS_data)

# Calculate the partial correlations
partial_corr_matrix = calculate_partial_correlations(expression_data)

# Print the full partial correlation matrix
print("Partial Correlation Matrix:")
print(partial_corr_matrix)

# Draw the network graph
draw_network_with_attribution_inhibition(partial_corr_matrix, gene_names, threshold=0.3)



# #Plot without calculating the correlation. Plot is based on predicted edges
# def plot_network(predicted_edges, gene_names, importance_threshold=0.25):
#     """
#     Visualizes the gene regulatory network (GRN) based on predicted edges.
#     :param predicted_edges: DataFrame with 'Gene1', 'Gene2', and 'Importance' columns
#     :param gene_names: List of gene names
#     """
#
#     # check if predicted_edges is not None
#     if predicted_edges is not None:
#         # filter predicted_edges for importance > 0.25
#         filtered_edges = predicted_edges[predicted_edges['Importance'] >= importance_threshold]
#
#         # Add edges with weights (Importance) from predicted edges
#         for index, row in filtered_edges.iterrows():
#             print(f"Gene1: {row['Gene1']}, Gene2: {row['Gene2']}, Importance: {row['Importance']}")
#         G = nx.DiGraph()
#
#             #G.add_edge(row['Gene1'], row['Gene2'], weight=row['Importance'])
#         for index, row in filtered_edges.iterrows():
#             # G.add_edge(row['Gene1'], row['Gene2'], weight=row['Importance'])
#             G.add_edge(row['Gene1'], row['Gene2'])
#
#         # Create a directed graph
#
#         # Add nodes for each gene
#         G.add_nodes_from(gene_names)
#
#
#         # Set up the plot
#         plt.figure(figsize=(12, 12))
#         pos = nx.spring_layout(G, k=0.15, iterations=20)  # Use spring layout for node placement
#
#         # Draw nodes and edges
#         # nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold',edge_color='gray', width=1, alpha=0.7)
#         nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=10, font_weight='bold',
#                 arrows=True)
#
#         # Draw edge weights (importance values)
#         edge_labels = nx.get_edge_attributes(G, 'weight')
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#
#         # Show the plot
#         plt.title("Predicted Gene Regulatory Network (GRN)")
#         plt.show()
#
#
# # After the GA loop and identifying the best individual:
# best_individual = hof[0]
# # print(f"Best individual: {best_individual}")
# # print(f"Fitness: {best_individual.fitness.values}")
#
# # Get the predicted network using the best hyperparameters
# xgb_kwargs_best = xgb_kwargs
#
# # Get the importance values using the best hyperparameters
# VIM_best = get_importances(TS_data, time_points, time_lag=0, gene_names=gene_names, regulators=regulators,
#                            alpha=0.022408670532763, SS_data=SS_data, param=xgb_kwargs_best)
#
# # Get predicted edges (GRN) from the best individual
# predicted_edges_best = get_links(VIM_best, gene_names, regulators, sort=True, file_name=None)
#
# # Plot the network for the best individual
# # plot_network(predicted_edges_best, gene_names)
# plot_network(predicted_edges_best, gene_names, importance_threshold=0.3)
#

