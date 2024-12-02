import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler


# Utility function to get the links (edges)
def get_links(VIM, gene_names, regulators, sort=True, file_name=None):
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score)
                  for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges, columns=['Gene1', 'Gene2', 'Importance'])
    if sort:
        pred_edges.sort_values(by='Importance', ascending=False, inplace=True)
    if file_name is None:
        return pred_edges
    else:
        pred_edges.to_csv(file_name, sep='\t', header=None, index=None)
        return None


# Function to compute the variable importance matrix (VIM)
def get_importances(TS_data, time_points, time_lag, gene_names, regulators, alpha=None, SS_data=None, param={}):
    time_start = time.time()
    ngenes = TS_data[0].shape[1]
    alphas = [alpha] * ngenes if alpha else [0.1] * ngenes  # Default alpha if not specified
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    VIM = np.zeros((ngenes, ngenes))

    for i in range(ngenes):
        input_idx = idx.copy()
        if i in input_idx:
            input_idx.remove(i)
        vi = get_importances_single(TS_data, time_points, time_lag, alphas[i], input_idx, i, SS_data)
        VIM[i, :] = vi

    time_end = time.time()
    print(f"Elapsed time: {time_end - time_start:.2f} seconds")
    return VIM


# Function to compute importance for a single target gene
def get_importances_single(TS_data, time_points, time_lag, alpha, input_idx, output_idx, SS_data):
    h = 1  # lag used for finite approximation of derivative
    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)
    nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data])
    ninputs = len(input_idx)

    # Construct learning sample
    input_matrix_time = np.zeros((nsamples_time - h * nexp, ninputs))
    output_vect_time = np.zeros(nsamples_time - h * nexp)
    nsamples_count = 0

    # Prepare time-series data
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

    # Steady-state data
    if SS_data is not None:
        input_matrix_steady = SS_data[:, input_idx]
        output_vect_steady = SS_data[:, output_idx] * alpha
        input_all = np.vstack([input_matrix_steady, input_matrix_time])
        output_all = np.concatenate((output_vect_steady, output_vect_time))
    else:
        input_all = input_matrix_time
        output_all = output_vect_time

    # Standardize the data (important for Lasso)
    scaler = StandardScaler()
    input_all = scaler.fit_transform(input_all)

    # Use LassoCV to automatically choose the best alpha using cross-validation
    lasso_cv = LassoCV(cv=5)  # 5-fold cross-validation
    lasso_cv.fit(input_all, output_all)
    alpha_best = lasso_cv.alpha_  # Get the optimal alpha
    print(f"Optimal alpha: {alpha_best}")

    # Fit the Lasso model with the optimal alpha
    lasso = Lasso(alpha=alpha_best)
    lasso.fit(input_all, output_all)

    feature_importances = lasso.coef_
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances

    return vi


# Compute AUROC and AUPR scores
def get_scores(VIM, gold_edges, gene_names, regulators):
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
    auroc = roc_auc_score(final['2_y'], final['2_x'])
    aupr = average_precision_score(final['2_y'], final['2_x'])
    return auroc, aupr


TS_data = pd.read_csv(
    "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_5_timeseries.tsv",
    sep='\t').values
SS_data_1 = pd.read_csv(
    "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_5_knockouts.tsv",
    sep='\t').values
SS_data_2 = pd.read_csv(
    "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_5_knockdowns.tsv",
    sep='\t').values

# Combine steady-state data
SS_data = np.vstack([SS_data_1, SS_data_2])

# Prepare time-series data
i = np.arange(0, 85, 21)
j = np.arange(21, 106, 21)
TS_data = [TS_data[i:j] for (i, j) in zip(i, j)]
time_points = [np.arange(0, 1001, 50)] * 5

# Set gene names and regulators
ngenes = TS_data[0].shape[1]
gene_names = ['G' + str(i + 1) for i in range(ngenes)]
regulators = gene_names.copy()

# Load gold edges (for scoring)
gold_edges = pd.read_csv(
    "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_5_goldstandard.tsv",
    '\t', header=None)


# Compute the variable importance matrix using LassoCV (with cross-validation)
VIM = get_importances(TS_data, time_points, time_lag=0, gene_names=gene_names, regulators=regulators, SS_data=SS_data)

# Compute AUROC and AUPR scores
auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)
print(f"AUROC: {auroc}, AUPR: {aupr}")

# Get the predicted network (links)
predicted_edges = get_links(VIM, gene_names, regulators, sort=True, file_name=None)
