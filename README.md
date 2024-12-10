# DHpT-GRNMOPT 

This project implements a method for predicting gene regulatory networks (GRNs) using time-series data, steady-state data, and a genetic algorithm (GA) for hyperparameter optimization of an XGBoost model.

## Overview

The goal of this project is to predict the interactions between genes by identifying regulatory relationships in gene expression data. This process involves:

- **Data Preprocessing**: Preprocessing time-series and steady-state data for model training.
- **XGBoost Model**: Using XGBoost to calculate feature importances based on gene interactions.
- **Genetic Algorithm for Hyperparameter Tuning**: Optimizing the hyperparameters of the XGBoost model using a genetic algorithm.
- **Network Prediction**: Using the optimized model to predict regulatory edges between genes.
- **Visualization**: Visualizing the resulting gene regulatory network using NetworkX and Matplotlib.

## Key Features

- **Hyperparameter Tuning with GA**: Optimizes XGBoost hyperparameters (`n_estimators`, `learning_rate`, and `max_depth`) using a genetic algorithm to enhance model performance.
- **Data Preprocessing**: Processes time-series and steady-state data for model training.
- **Model Training**: Uses XGBoost to compute feature importances that indicate gene regulatory interactions.
- **Network Prediction**: Identifies regulatory edges between genes based on feature importances.
- **Visualization**: Visualizes the predicted gene regulatory network using NetworkX and Matplotlib.

## Hyperparameter Optimization with Genetic Algorithm

In this project, the **genetic algorithm (GA)** is used to optimize the hyperparameters of the XGBoost model. Specifically, the GA tunes the following hyperparameters:

- `n_estimators`: Number of boosting rounds.
- `learning_rate`: Learning rate for the model.
- `max_depth`: Maximum depth of the individual trees in the XGBoost model.

The GA evaluates multiple sets of hyperparameters and selects the ones that yield the best model performance, measured using metrics like **AUROC** (Area Under the Receiver Operating Characteristic Curve) and **AUPR** (Area Under the Precision-Recall Curve).

The GA operates in the following way:

1. **Population Initialization**: A population of individuals, each representing a set of hyperparameters, is generated.
2. **Selection**: The best individuals are selected based on their fitness, which is determined by the model's performance.
3. **Crossover**: Selected individuals are paired, and their hyperparameters are recombined to form offspring.
4. **Mutation**: Offspring undergo mutation, where their hyperparameters are randomly adjusted.
5. **Evaluation**: The fitness of the new individuals is evaluated using the selected hyperparameters and model performance.
6. **Elitism**: The best-performing individuals are retained for the next generation.

This process is repeated for several generations, with the goal of evolving the best set of hyperparameters for the XGBoost model.


## Usage
**Step-by-Step Process** 

1. Prepare the Data: Place your time-series and steady-state data in the appropriate format as input files.
2. Run the Genetic Algorithm: The genetic algorithm will optimize the hyperparameters for the XGBoost model. It will tune parameters like n_estimators, learning_rate, and max_depth to maximize model performance.
3. Predict the Gene Regulatory Network: After hyperparameter optimization, the best model will be used to calculate feature importances, which will be used to predict regulatory edges between genes.
4. Visualize the Network: The predicted gene regulatory network will be visualized using NetworkX and Matplotlib.
5. For more detailed test, please refer to the /DHpT_GRNMOPT.



## Example Workflow
**Load Data** : Load the time-series and steady-state data files.
**Run GA** : The genetic algorithm will optimize the hyperparameters of the XGBoost model.
**Train Model** : Using the optimized hyperparameters, the XGBoost model will calculate feature importances (VIM).
**Predict Network** : The regulatory network (GRN) will be predicted based on the feature importances.
**Visualize** : The resulting network will be visualized using NetworkX.


## Installation

To install the required dependencies, run the following command:

```bash
pip install deap xgboost numpy pandas scikit-learn networkx matplotlib
```

### Contribution
Feel free to contribute to the project by creating issues or submitting pull requests. To submit a pull request, please fork the repository, make your changes, and submit a pull request.
