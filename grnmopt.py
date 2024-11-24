from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, average_precision_score
import geatpy as ea
class MyProblem(ea.Problem):
    def __init__(self):
        name = 'MyProblem'
        M = 2 # Initialize M (target dimension)
        maxormins = [-1] * M
        Dim = 2  # Initialize Dim (dimension of decision variable)
        varTypes = [0,1]   # The type of decision variable 0: variable is continuous  1: variable is discrete
        lb = [0,0]        # Lower bound of decision variable
        ub = [1,3]        # Upper bound of decision variable
        lbin = [1,1]    # 1 indicates that the lower boundary can be obtained
        ubin = [1,1]    #  1 indicates that the upper boundary can be obtained

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    def aimFunc(self, pop): # objective function
        Objv1=[]
        Objv2=[]
        alphas = pop.Phen[:,0]
        time_lag = pop.Phen[:,1]
        pop.ObjV = np.zeros((pop.Phen.shape[0], self.M))

        for (alpha,time_lag) in zip(alphas,time_lag):
            time_lag = (int)(time_lag)
            def get_links(VIM, gene_names, regulators, sort=True, file_name=None):

                idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
                pred_edges = [(gene_names[j], gene_names[i], score) for (i, j),score in np.ndenumerate(VIM) if i!=j and j in idx]
                pred_edges = pd.DataFrame(pred_edges)
                if sort is True:
                    pred_edges.sort_values(2, ascending=False, inplace=True)
                if file_name is None:
                    print(pred_edges)
                else:
                    pred_edges.to_csv(file_name, sep='\t', header=None, index=None)

            def get_importances(TS_data, time_points, time_lag, gene_names,regulators, alpha, SS_data=None, param={}):

                time_start = time.time()

                ngenes = TS_data[0].shape[1]

                alphas = [alpha] * ngenes

                # Get the indices of the candidate regulators
                idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

                # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
                VIM = np.zeros((ngenes,ngenes))

                for i in range(ngenes):
                    input_idx = idx.copy()
                    if i in input_idx:
                        input_idx.remove(i)
                    vi = get_importances_single(TS_data, time_points, time_lag, alphas[i], input_idx, i, SS_data, param)
                    VIM[i,:] = vi

                time_end = time.time()
                print("Elapsed time: %.2f seconds" % (time_end - time_start))

                return VIM

            def get_importances_single(TS_data, time_points, time_lag, alpha, input_idx, output_idx, SS_data, param):

                h = 1 # lag (in number of time points) used for the finite approximation of the derivative of the target gene expression
                ngenes = TS_data[0].shape[1]
                nexp = len(TS_data)
                nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data])
                ninputs = len(input_idx)
                # Construct learning sample
                # Time-series data
                input_matrix_time = np.zeros((nsamples_time-h*nexp,ninputs))
                output_vect_time = np.zeros(nsamples_time-h*nexp)
                nsamples_count = 0
                for (i,current_timeseries) in enumerate(TS_data):
                    current_time_points = time_points[i]
                    npoints = current_timeseries.shape[0]
                    time_diff_current = current_time_points[h:] - current_time_points[:npoints-h]
                    current_timeseries_input = current_timeseries[:npoints-h,input_idx]
                    current_timeseries_output = (current_timeseries[h:,output_idx] - current_timeseries[:npoints-h,output_idx]) / time_diff_current + alpha*current_timeseries[:npoints-h,output_idx]


                    #Time delay
                    npoints = current_timeseries_input.shape[0]
                    current_timeseries_input = current_timeseries_input[:npoints - time_lag]
                    current_timeseries_output = current_timeseries_output[time_lag:]

                    # print(f"Original npoints: {npoints}")
                    # print(f"Value of h: {h}")
                    # print(f"Length of current_time_points: {len(current_time_points)}")
                    #
                    # print(f"Original length of current_time_points: {len(current_time_points)}")
                    # print(f"Length of h slice (current_time_points[h:]): {len(current_time_points[h:])}")
                    # print(
                    #     f"Length of npoints-h slice (current_time_points[:npoints - h]): {len(current_time_points[:npoints - h])}")
                    # print(f"Value of npoints: {npoints}")
                    # print(f"Value of h: {h}")

                    nsamples_current = current_timeseries_input.shape[0]
                    input_matrix_time[nsamples_count:nsamples_count+nsamples_current,:] = current_timeseries_input
                    output_vect_time[nsamples_count:nsamples_count+nsamples_current] = current_timeseries_output
                    nsamples_count += nsamples_current

                # Steady-state data
                if SS_data is not None:
                    input_matrix_steady = SS_data[:,input_idx]
                    output_vect_steady = SS_data[:,output_idx] * alpha

                    # Concatenation
                    input_all = np.vstack([input_matrix_steady,input_matrix_time])
                    output_all = np.concatenate((output_vect_steady,output_vect_time))
                else:
                    input_all = input_matrix_time
                    output_all = output_vect_time

                treeEstimator = XGBRegressor(**param)

                # Learn ensemble of trees
                treeEstimator.fit(input_all,output_all)

                # Compute importance scores
                feature_importances = treeEstimator.feature_importances_
                vi = np.zeros(ngenes)
                vi[input_idx] = feature_importances

                return vi

            def get_scores(VIM, gold_edges, gene_names, regulators):

                idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
                pred_edges = [(gene_names[j], gene_names[i], score) for (i, j), score in np.ndenumerate(VIM) if i != j and j in idx]
                pred_edges = pd.DataFrame(pred_edges)
                #pred_edges = pred_edges.iloc[:100000]
                final = pd.merge(pred_edges, gold_edges, on=[0, 1], how='inner')
                auroc = roc_auc_score(final['2_y'], final['2_x'])
                aupr = average_precision_score(final['2_y'], final['2_x'])
                return auroc, aupr

            TS_data = pd.read_csv("/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_timeseries.tsv", sep='\t').values
            SS_data_1 = pd.read_csv("/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_knockouts.tsv", sep='\t').values
            SS_data_2 = pd.read_csv("/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_knockdowns.tsv", sep='\t').values


            # get the steady-state data
            SS_data = np.vstack([SS_data_1, SS_data_2])

            i = np.arange(0, 85, 21)
            j = np.arange(21, 106, 21)

            # get the time-series data
            TS_data = [TS_data[i:j] for (i, j) in zip(i, j)]
            # get time points
            time_points = [np.arange(0, 1001, 50)] * 5

            ngenes = TS_data[0].shape[1]
            gene_names = ['G'+str(i+1) for i in range(ngenes)]
            regulators = gene_names.copy()

            gold_edges = pd.read_csv("/home/mourinho/Desktop/probability models/project/GRNMOPT/DREAM4/insilico_size10/insilico_size10_1_goldstandard.tsv", '\t', header=None)



            xgb_kwargs = dict(n_estimators=398, learning_rate=0.0137089260215423, importance_type="weight", max_depth=5, n_jobs=-1,objective ='reg:squarederror')

            VIM = get_importances(TS_data, time_points,  time_lag=time_lag, gene_names=gene_names, regulators=regulators,
                                    alpha = alpha,SS_data=SS_data, param=xgb_kwargs)
            auroc, aupr = get_scores(VIM, gold_edges, gene_names, regulators)

            Objv1.append(auroc)
            Objv2.append(aupr)

        pop.ObjV[:,[0]] = np.array([Objv1]).T
        pop.ObjV[:,[1]] = np.array([Objv2]).T


if __name__ == "__main__":
    problem = MyProblem()
    Encoding = 'RI'           # Encoding method
    NIND = 10               # Gen
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)
    myAlgorithm.MAXGEN = 10
    myAlgorithm.logTras = 0
    myAlgorithm.verbose = False
    myAlgorithm.drawing = 0
    [NDSet, population] = myAlgorithm.run()
    NDSet.save()
    print('用时：%s 秒' % myAlgorithm.passTime)
    print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
    print(NDSet.ObjV)
    print(NDSet.Phen)


