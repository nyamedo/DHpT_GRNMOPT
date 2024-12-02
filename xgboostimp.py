import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBRegressor
import geatpy as ea


# Helper functions for feature importance and scoring (to be added below)
def get_importances(TS_data, time_points, time_lag, gene_names, regulators, alpha, SS_data=None, param={}):
    time_start = time.time()
    ngenes = TS_data[0].shape[1]
    alphas = [alpha] * ngenes

    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
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


def get_scores(VIM, gold_edges, gene_names, regulators):
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
    pred_edges = pd.DataFrame(pred_edges)
    final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
    auroc = roc_auc_score(final['2_y'], final['2_x'])
    aupr = average_precision_score(final['2_y'], final['2_x'])
    return auroc, aupr


class MyProblem(ea.Problem):
    def __init__(self):
        name = 'HyperparameterTuningXGB'
        M = 2  # Number of objectives: AUROC and AUPR
        maxormins = [-1] * M  # Maximize both objectives
        Dim = 5  # Decision variables: alpha, time_lag, n_estimators, learning_rate, max_depth
        varTypes = [0, 1, 1, 0, 0]  # Continuous for alpha, time_lag, n_estimators, learning_rate, max_depth
        lb = [0, 0, 10, 0.01, 1]  # Lower bounds for each variable
        ub = [1, 3, 500, 0.5, 15]  # Upper bounds for each variable
        lbin = [1, 1, 1, 1, 1]  # Can reach the lower bound
        ubin = [1, 1, 1, 1, 1]  # Can reach the upper bound

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Objv1 = []  # AUROC
        Objv2 = []  # AUPR
        alphas = pop.Phen[:, 0]  # Extract alpha values
        time_lag = pop.Phen[:, 1]  # Extract time_lag values
        n_estimators = pop.Phen[:, 2].astype(int)  # Extract n_estimators (discrete)
        learning_rate = pop.Phen[:, 3]  # Extract learning_rate values
        max_depth = pop.Phen[:, 4].astype(int)  # Extract max_depth (discrete)

        pop.ObjV = np.zeros((pop.Phen.shape[0], self.M))

        for alpha, tlag, n_estim, lr, mdepth in zip(alphas, time_lag, n_estimators, learning_rate, max_depth):
            tlag = int(tlag)
            # Define the XGBoost hyperparameters
            xgb_kwargs = {
                'n_estimators': n_estim,
                'learning_rate': lr,
                'max_depth': mdepth,
                'importance_type': 'weight',
                'n_jobs': -1,
                'objective': 'reg:squarederror'
            }

            # Get feature importances and scores
            VIM = get_importances(TS_data, time_points, time_lag=tlag, gene_names=gene_names,
                                  regulators=regulators, alpha=alpha, SS_data=SS_data, param=xgb_kwargs)
            auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)

            Objv1.append(auroc)
            Objv2.append(aupr)

        # Assign objectives to the population
        pop.ObjV[:, [0]] = np.array([Objv1]).T
        pop.ObjV[:, [1]] = np.array([Objv2]).T


if __name__ == "__main__":
    # Loading necessary data
    TS_data = pd.read_csv(
        "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_timeseries.tsv",
        sep='\t').values
    SS_data_1 = pd.read_csv(
        "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_knockouts.tsv",
        sep='\t').values
    SS_data_2 = pd.read_csv(
        "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_knockdowns.tsv",
        sep='\t').values

    SS_data = np.vstack([SS_data_1, SS_data_2])

    i = np.arange(0, 85, 21)
    j = np.arange(21, 106, 21)
    TS_data = [TS_data[i:j] for i, j in zip(i, j)]
    time_points = [np.arange(0, 1001, 50)] * 5

    ngenes = TS_data[0].shape[1]
    gene_names = ['G' + str(i + 1) for i in range(ngenes)]
    regulators = gene_names.copy()

    gold_edges = pd.read_csv(
        "/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_goldstandard.tsv",
        '\t', header=None)

    # Initialize GA problem
    problem = MyProblem()

    # Set the population size and other parameters
    Encoding = 'RI'  # Real-valued encoding
    NIND = 20  # Population size
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)

    # Define the algorithm
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)
    myAlgorithm.MAXGEN = 50  # Number of generations
    myAlgorithm.logTras = 1  # Enable logging for debugging
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 1  # Enable drawing the result

    # Run the algorithm
    [NDSet, population] = myAlgorithm.run()

    # Save and print results
    NDSet.save()
    print('Time elapsed: %s seconds' % myAlgorithm.passTime)
    print('Number of non-dominated solutions: %d' % NDSet.sizes if NDSet.sizes != 0 else 'No feasible solution found!')
    print(NDSet.ObjV)  # Display the Pareto front (AUROC, AUPR)
    print(NDSet.Phen)  # Display the hyperparameters corresponding to the Pareto front
