from typing import List, Dict, Callable, Union, Optional

import numpy as np

def exact(high_frequency_data: ArrayLike, 
           low_frequency_data: ArrayLike, 
           annotation_data: ArrayLike, 
           linear_constraint: ArrayLike, 
           quadratic_constraint_weight: Union[float,int]) -> ArrayLike:

    t = len(low_frequency_data)
    T = len(high_frequency_data)
    k = T // t

    constraints_matrix = np.kron(np.eye(t), linear_constraint.reshape(-1,1))

    cost_derivative_matrix = np.eye(T)
    diag_idx0, diag_idx1 = np.diag_indices(T)
    cost_derivative_matrix[diag_idx0[1:-1], diag_idx1[1:-1]] += 1
    cost_derivative_matrix[diag_idx0[:-1]+1, diag_idx1[:-1]] = -1
    cost_derivative_matrix[diag_idx0[:-1], diag_idx1[:-1]+1] = -1

    cost_matrix = 2 * cost_derivative_matrix

    annotation_matrix = np.zeros(annotation_data.shape)
    annotation_matrix[~np.isnan(annotation_data)] = 1
    annotation_matrix = np.diag(annotation_matrix)
    annotation_matrix = 2 * quadratic_constraint_weight * annotation_matrix

    matrix = np.zeros((T+t, T+t))  
    matrix[:T,:T] = cost_matrix + annotation_matrix
    matrix[:T,T:] = constraints_matrix
    matrix[T:,:T] = constraints_matrix.T

    values = np.zeros((t+T,1)) 
    values[:T] = (np.hstack([0, np.diff(high_frequency_data)]) - np.hstack([np.diff(high_frequency_data), 0])).reshape(-1,1)
    values[:T] += 2 * quadratic_constraint_weight * np.nan_to_num(annotation_data).reshape(-1,1)
    values[-t:] = low_frequency_data.reshape(-1,1) 

    X = np.linalg.solve(matrix, values)
    X = X[:-t] 

    return X.squeeze()

# def inexact