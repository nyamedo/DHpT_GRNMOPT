import random
from deap import base, creator, tools
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, squareform
from matplotlib.patches import FancyArrowPatch

# Creates and returns a data frame of predicted edges when given importance scores, gene_names, and regulators
def get_links(VIM, gene_names, regulators, sort=True, file_name=None):
    #Identify indices of genes that are regulators
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

    # Create a list of predicted edges where regulator genes influence other genes
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

# Computes importance scores for each gene using time-series and steady-state data.
# Returns an n by n matrix of importance scores
def get_importances(TS_data, time_points, time_lag, gene_names, regulators, alpha, SS_data=None, param={}):
    # times the function execution
    time_start = time.time() 

    # number of genes in teh data set
    ngenes = TS_data[0].shape[1]

    # Creates a list of regularization parameters for each gene
    alphas = [alpha] * ngenes

    # Get the indices of the candidate regulators
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

    # Initializes the importance score matrix to all zeroes
    VIM = np.zeros((ngenes, ngenes))
    
    # Compute each gene's importance score by iterating through each gene
    for i in range(ngenes):
        input_idx = idx.copy() # makes a copy of the regulator indices
        if i in input_idx:
            input_idx.remove(i) # prevents self-regulation
        # Computes the importance score for each gene
        vi = get_importances_single(TS_data, time_points, time_lag, alphas[i], input_idx, i, SS_data, param)
        VIM[i, :] = vi # Add the importance score to the matrix

    # End timing of function
    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    # return the computed importance matrix
    return VIM

# Computes the importance of each regulatory gene on a target gene using XGBoost regression.
# Returns vi, an importance score for each regulator gene
def get_importances_single(TS_data, time_points, time_lag, alpha, input_idx, output_idx, SS_data, param):
    h = 1  # lag (in number of time points) used for the finite approximation of the derivative of the target gene expression
    ngenes = TS_data[0].shape[1] # Number of genes in dataset
    nexp = len(TS_data) # Number of experiments to run
    nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data]) # Total number of time points across all experiments
    ninputs = len(input_idx) # Number of candidate regulators

    # initialize all the training data matrices
    input_matrix_time = np.zeros((nsamples_time - h * nexp, ninputs))
    output_vect_time = np.zeros(nsamples_time - h * nexp)
    
    nsamples_count = 0 # Tracks number of samples

    # Loops through each time-series experiment to construct training samples
    for (i, current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i] # Time points for current experiment
        npoints = current_timeseries.shape[0] # Number of time points in the current experiment
        # Computes the time difference between successive poitns
        time_diff_current = current_time_points[h:] - current_time_points[:npoints - h] 
        # Extracts expression values for candidate regulators
        current_timeseries_input = current_timeseries[:npoints - h, input_idx]
        # Computes the approximate derivative of the targe gene's expression
        current_timeseries_output = (current_timeseries[h:, output_idx] - current_timeseries[:npoints - h,
                                                                          output_idx]) / time_diff_current + alpha * current_timeseries[
                                                                                                                     :npoints - h,
                                                                                                                     output_idx]

        # Time delay adjustment
        npoints = current_timeseries_input.shape[0]
        current_timeseries_input = current_timeseries_input[:npoints - time_lag]
        current_timeseries_output = current_timeseries_output[time_lag:]

        # Stores processed input and output data
        nsamples_current = current_timeseries_input.shape[0]
        input_matrix_time[nsamples_count:nsamples_count + nsamples_current, :] = current_timeseries_input
        output_vect_time[nsamples_count:nsamples_count + nsamples_current] = current_timeseries_output
        nsamples_count += nsamples_current

    # Steady-state data
    # If available, combine it with the time-series data
    if SS_data is not None:
        input_matrix_steady = SS_data[:, input_idx]
        output_vect_steady = SS_data[:, output_idx] * alpha

        # Concatenation
        input_all = np.vstack([input_matrix_steady, input_matrix_time])
        output_all = np.concatenate((output_vect_steady, output_vect_time))
    else:
        input_all = input_matrix_time
        output_all = output_vect_time

    # Train an XGBoost regressor on the extracted features
    treeEstimator = XGBRegressor(**param)
    # Learn ensemble of trees and fits regressor
    treeEstimator.fit(input_all, output_all)

    # Compute importance scores
    feature_importances = treeEstimator.feature_importances_
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances # Assigns importance scores to the corresponding gene

    return vi # Return importance scores

# Calculates the AUROC and AUPR scores based on predicted and actual gene regulatory edges
def get_scores(VIM, gold_edges, gene_names, regulators):
    # Identify indices that are regulator genes
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    # Get predicted edges and scores and convert to a data frame
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    # Merge predicted edges with ground proof edges to evaluate performance
    final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
    # Get auroc and aupr scores then return them
    auroc = roc_auc_score(final['2_y'], final['2_x'])
    aupr = average_precision_score(final['2_y'], final['2_x'])
    return auroc, aupr

# Load in time series and steady state data
TS_data = pd.read_csv(
    "DREAM4/insilico_size10/insilico_size10_1_timeseries.tsv",
    sep='\t').values
SS_data_1 = pd.read_csv(
    "DREAM4/insilico_size10/insilico_size10_1_knockouts.tsv",
    sep='\t').values
SS_data_2 = pd.read_csv(
    "DREAM4/insilico_size10/insilico_size10_1_knockdowns.tsv",
    sep='\t').values

# SS_data_3 = pd.read_csv("/Users/macbookpro/Documents/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_multifactorial.tsv",sep='\t').values

# get the steady-state data
SS_data = np.vstack([SS_data_1, SS_data_2])

# Define time indices for splitting time series data
i = np.arange(0, 85, 21)
j = np.arange(21, 106, 21)

# get the time-series data
TS_data = [TS_data[i:j] for (i, j) in zip(i, j)]
# get time points
time_points = [np.arange(0, 1001, 50)] * 5

# Get the gene names and ground truth regulators
ngenes = TS_data[0].shape[1]
gene_names = ['G' + str(i + 1) for i in range(ngenes)]
regulators = gene_names.copy()

# Read in the gold standard edges
gold_edges = pd.read_csv(
    "DREAM4/insilico_size10/insilico_size10_1_goldstandard.tsv",
    sep = '\t', header=None)

# Define hyperparameters for XGBoost algorithms
xgb_kwargs = dict(n_estimators=398, learning_rate=0.0137089260215423, importance_type="weight", max_depth=5, n_j123obs=-1,
                  objective='reg:squarederror')

'''# Compute the regulatory importance matrix
VIM = get_importances(TS_data, time_points, time_lag=0, gene_names=gene_names, regulators=regulators,
                      alpha=0.022408670532763, SS_data=SS_data, param=xgb_kwargs)
# Evaluate network by getting auroc and aupr scores
auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)
# print(auroc, aupr)

# get_links to get the predicted network
predicted_edges = get_links(VIM, gene_names, regulators, sort=True, file_name=None)'''

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

'''def get_importances_single(TS_data, time_points, time_lag, alpha, input_idx, output_idx, SS_data, param):
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
#     return auroc, aupr'''


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


# Define the fitness function as a multi-objective optimization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))  # Maximize both AUROC and AUPR
creator.create("Individual", list, fitness=creator.FitnessMax) # Each individual represents a set of hyperparameters

# Initialize a DEAP toolbox to define genetic operators
toolbox = base.Toolbox()

# Define the bounds for the hyperparameters
BOUNDS_LOW = [50, 0.001, 3]  # n_estimators: [50, 500], learning_rate: [0.001, 0.1], max_depth: [3, 15]
BOUNDS_HIGH = [500, 0.1, 15]

# Register the creation of individuals with random values within bounds
for i in range(3):  # We have 3 hyperparameters
    toolbox.register(f"hyperparameter_{i}", random.uniform, BOUNDS_LOW[i], BOUNDS_HIGH[i])

# Create random individuals based on the hyperparameter bounds
toolbox.register("individualCreator", tools.initCycle, creator.Individual,
                 tuple([toolbox.__getattribute__(f"hyperparameter_{i}") for i in range(3)]), n=1)

# Create an individual with the selected hyperparameters
toolbox.register("population", tools.initRepeat, list, toolbox.individualCreator)

# Register crossover, mutation, and selection methods
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=1.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUNDS_LOW, up=BOUNDS_HIGH, eta=1.0, indpb=1.0/3)
toolbox.register("select", tools.selTournament, tournsize=3)

# Register the evaluation function
toolbox.register("evaluate", evaluate, TS_data=TS_data, time_points=time_points, time_lag=0, gene_names=gene_names,
                 regulators=regulators, SS_data=SS_data, gold_edges=gold_edges)

# Set Genetic Algorithm parameters
population_size = 30
generations = 2
cx_probability = 0.7
mut_probability = 0.3

# Create an initial population
population = toolbox.population(n=population_size)

# Create a Hall of Fame to store the best individual found during evolution
hof = tools.HallOfFame(1)

# Define the statistics (optional)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Starts the genetic algorithm loop
# Run the genetic algorithm with elitism using eaSimple
for gen in range(generations):
    print(f"Generation {gen}")

    # Evaluate fitness for all individuals in the population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit # Assign fitness values to individuals

    # Select the best individuals for reproduction
    selected = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, selected))

    # Apply crossover 
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx_probability:
            toolbox.mate(child1, child2) # Perform crossover
            del child1.fitness.values # Invalidate fitness
            del child2.fitness.values # Invalidate fitness

    # Apply mutation to offspring
    for mutant in offspring:
        if random.random() < mut_probability:
            toolbox.mutate(mutant) # Perform mutation
            del mutant.fitness.values # Invalidate fitness

    # Recalculate fitness for offspring with invalidated fitness values
    for ind in offspring:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    # Update the Hall of Fame with the best individual
    hof.update(offspring)

    # Replace the old population with the new one
    population[:] = offspring

    # Log statistics for the current generation
    record = stats.compile(population)
    print(record)

# After running GA, output the best individual
best_individual = hof[0]
print(f"Best individual: {best_individual}")
print(f"Fitness: {best_individual.fitness.values}")


# Calculate the partial correlation matrix, which measures the direct relationships 
# between variables while controlling for the influence of other variables.
def calculate_partial_correlations(data_matrix):
    
    # Calculate the covariance matrix
    covariance_matrix = np.cov(data_matrix, rowvar=False)
    # take the inverse of the covariance matrix to compute the precision matrix
    precision_matrix = np.linalg.inv(covariance_matrix)

    # Get diagonal elements for normalization
    d = np.sqrt(np.diag(precision_matrix))
    # Compute partial correlation matrix using normalization
    partial_corr_matrix = -precision_matrix / np.outer(d, d)

    # Set the diagonal elements to 1 since self-correlation is always 1
    np.fill_diagonal(partial_corr_matrix, 1)

    return partial_corr_matrix

# Adjust node positions in a graph to ensure a minimum distance between them, preventing overlap.
def adjust_node_positions(pos, min_distance=3, max_iterations=100):
    # Get node labels
    nodes = list(pos.keys())
    positions = np.array([pos[n] for n in nodes]) # Convert positions to numpy array
    
    for _ in range(max_iterations):
        # Compute pairwise distances between all nodes, ignoring self-distances
        distances = squareform(pdist(positions))  
        np.fill_diagonal(distances, np.inf)  

        # Identify pairs of nodes that are too close to each other
        overlap_indices = np.where(distances < min_distance)
        if len(overlap_indices[0]) == 0:
            break  # Exit early if no overlaps are found

        # Adjust position of overlapping nodes
        for i, j in zip(*overlap_indices):
            if i < j:  # Avoid adjusting the same pair twice
                vec = positions[j] - positions[i] # Compute direction vector
                norm = np.linalg.norm(vec) # Computer distance

                # If nodes are in the same place, apply a small random shift
                if norm == 0:
                    vec = np.random.rand(2) - 0.5  # Random small shift if nodes are exactly on top
                    norm = np.linalg.norm(vec)
                # Move nodes apart to maintain the minimum distance
                displacement = (min_distance - norm) * (vec / norm) * 0.5
                positions[i] -= displacement
                positions[j] += displacement
                
    # Convert back to dict and return nodes
    return {nodes[i]: tuple(positions[i]) for i in range(len(nodes))}

#Draws directional arrows on a graph, distinguishing between positive and negative correlations.
def draw_custom_arrows(ax, pos, G, arrow_length = 0.12):

    for u, v in G.edges():
        # Get the weight (correlation value) of the edge, defaulting to 0 if not defined
        corr = G[u][v].get('weight', 0) 
        
        # Adjust arrowhead style and color based on correlation sign
        if corr > 0:
            arrowhead = "-|>"  # Positive correlation (right arrowhead)
            color = "black"    # Black for positive correlation
        else:
            arrowhead = "-["   # Negative correlation (-[ arrowhead)
            color = "red"      # Red for negative correlation

        # Get the start and end positions of the arrow
        start_pos = np.array(pos[u])
        end_pos = np.array(pos[v])

        # Calculate direction vector from start to end node
        direction = end_pos - start_pos
        distance = np.linalg.norm(direction)
        
        # Normalize the direction vector and adjust the endpoint to avoid overlap
        direction /= distance  # Normalize to unit vector
        shortened_end_pos = end_pos - direction * arrow_length

        # Create and add the arrow patch to the plot
        arrow = FancyArrowPatch(
            posA=start_pos, posB=shortened_end_pos,
            arrowstyle=arrowhead, color=color,
            mutation_scale=15, lw=2,
            connectionstyle="arc3,rad=0.1"  # Slight curve for better visibility
        )
        ax.add_patch(arrow)

# Draws a directed gene regulatory network based on partial correlation values, distinguishing between positive and negative correlations.
def draw_network_with_attribution_inhibition(correlation_matrix, gene_names, threshold=0.25):

    # Create a directed graph
    G = nx.DiGraph()  # Use a directed graph to represent gene regulation
    filtered_edges = predicted_edges_best[predicted_edges_best['Importance'] >= threshold] # Filter edges by threshold

    # Add nodes to the graph (even for genes that might not have connections)
    G.add_nodes_from(gene_names)

    # Remove isolated nodes (genes with no edges)
    G.remove_nodes_from(list(nx.isolates(G)))  

    # Add edges based on partial correlation threshold
    for i in range(len(filtered_edges)):
        gene1 = filtered_edges.iloc[i]['Gene1'] # Source gene
        gene2 = filtered_edges.iloc[i]['Gene2'] # Target gene
        importance = filtered_edges.iloc[i]['Importance'] # Importance score (weight)

        # Retrieve correlation value from the correlation matrix
        corr = correlation_matrix[gene_names.index(gene1), gene_names.index(gene2)]

        # Add directed edge: positive correlation -> gene1 regulates gene2
        if corr > 0:
            G.add_edge(gene1, gene2, weight=corr)  
        elif corr < 0:
            G.add_edge(gene2, gene1, weight=corr)  # Reverse direction for inhibition (negative correlation)

    # Circular layout for better visualization
    pos = nx.circular_layout(G)

    # Adjust node positions to prevent overlap
    pos = adjust_node_positions(pos)

    # Create a figure and axis for visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='black', linewidths=1.5, node_size=2000, ax=ax)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    # Custom arrow drawing
    draw_custom_arrows(ax, pos, G)

    # Add a legend to indicate positive (attribution) and negative (inhibition) relationships
    plt.scatter([], [], color='black', label="Attribution (+)")
    plt.scatter([], [], color='red', label="Inhibition (-)")
    plt.legend(loc="upper right")
    
    plt.title(f"Gene Network (Threshold: {threshold})")
    plt.show()

#Processes time-series gene expression data, computes partial correlations, and visualizes the network at each time step.
def process_time_series(time_series_data, gene_names, threshold=0.2):
    num_time_steps = len(time_series_data)  # Get number of time steps
    
    for t, data_matrix in enumerate(time_series_data):
        print(f"Processing time step {t+1}/{num_time_steps}")
        
        # Compute partial correlation matrix for the current time step
        partial_corr_matrix = calculate_partial_correlations(data_matrix)
        
        # Visualize the gene network at this time step
        draw_network_with_attribution_inhibition(partial_corr_matrix, gene_names, threshold)


# After the genetic algorithm (GA) loop, retrieve the best individual solution
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

expression_data = np.vstack(TS_data)

# Calculate the partial correlations
partial_corr_matrix = calculate_partial_correlations(expression_data)

# Print the full partial correlation matrix
print("Partial Correlation Matrix:")
print(partial_corr_matrix)

# Draw the network graph
draw_network_with_attribution_inhibition(partial_corr_matrix, gene_names, threshold=0.2) 

process_time_series(TS_data, gene_names)
