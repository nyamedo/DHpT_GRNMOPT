# GRNMOPT: Inference of Gene Regulatory Networks Based on A Multi-objective Optimization Approach


1 College of Information Science and Technology, Dalian Maritime University, Dalian 116039, China



**GRNMOPT is a scalable method exploiting utilizes decay rate and time-delay together to construct ODEs model, a multi-objective optimization method based on the Non-dominated Sorting Genetic Algorithms II (NSGA-II) is applied in GRNMOPT to simultaneously optimize decay rate and time-delay.** 

If you find our method is useful, please cite our paper:


### The describe of the program 

```
The program of grnmopt is to simultaneously optimize decay rate and time-delay.

The program of case_size10_net1 is to infer GRNs in specific decay rate and time-delay.
```



### The version of Python and packages
    Python version=3.7.4
    geatpy version 2.7.0
    Xgboost version=1.6.1
    scikit-learn version=1.0.2
    numpy version=1.21.6
    
### Parameters
    grnmopt:
        Objv: a list of objective function
        alpha:a constant of gene decay rate
        time_lag: a integer of gene time-delay
        NIND: a size of iterations
        param: a dict of parameters of xgboost
    	
    case_size10_net1:
        TS_data: a matrix of time-series data
        time_points: a list of time points
        SS_data: a matrix of time-series data, the default is "none"
        gene_names: a list of gene names
        regulators: a list of names of regulatory genes, the default is "all", 
        param: a dict of parameters of xgboost



