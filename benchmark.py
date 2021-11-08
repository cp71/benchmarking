from typing import List, Dict, Callable, Union, Optional

import numpy as np
from numpy.typing import ArrayLike

from sklearn.cross_decomposition import CCA
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.utils.validation as sklearn_validation

class Estimator(BaseEstimator, TransformerMixin):
    def __init__(self, linear_constraint: Optional[ArrayLike] = None, quadratic_constraint_weight: Optional[Union[float,int]] = None, solver: Optional[Callable] = None):
        self.linear_constraint = linear_constraint
        self.quadratic_constraint_weight = quadratic_constraint_weight
        self.solver = solver
         
    def fit(self, high_frequency_data: ArrayLike, low_frequency_data: ArrayLike, annotation_data: Optional[ArrayLike] = None):
        self.high_frequency_data = high_frequency_data
        self.low_frequency_data = low_frequency_data
        self.annotation_data = annotation_data
        
        self._validate()
        
        if self.solver is None:
            self.interpolant = self._solver() 
        else:
            self.interpolant = self.solver(self.high_frequency_data, self.low_frequency_data, self.annotation_data, self.linear_constraint, self.quadratic_constraint_weight) 
        
        return self

    def transform(self, X=None):
        
        model = CCA(n_components=1)
        X_c, Y_c = model.fit_transform(self.high_frequency_data.reshape(-1,self.frequency), self.low_frequency_data.reshape(-1,1))

        self.projection = model.x_rotations_.flatten()
        return np.squeeze(X_c)
    
    def predict(self, X = None):
        sklearn_validation.check_is_fitted(self, 'interpolant')
        return self.interpolant
    
    def score(self, X = None, y = None):
        sklearn_validation.check_is_fitted(self, 'interpolant')
        model = CCA(n_components=1)
        X_c, Y_c = model.fit_transform(self.interpolant.reshape(-1,self.frequency), self.low_frequency_data.reshape(-1,1))
        return np.corrcoef(np.squeeze(X_c), np.squeeze(Y_c))[0,1]
        
    def _validate(self):
        # high_frequency_data and low_frequency_data
        sklearn_validation.check_array(self.high_frequency_data, 
                                       accept_sparse = False, 
                                       dtype = "numeric", 
                                       force_all_finite=True, 
                                       ensure_2d=False, 
                                       allow_nd=False);

        self.high_frequency_data = np.asarray(self.high_frequency_data)

        if (self.high_frequency_data.ndim == 2):
            self.high_frequency_data = self.high_frequency_data.flatten()    

        if (self.solver is None) and np.any(np.isclose(self.high_frequency_data, 0)):
            raise ZeroDivisionError("high frequency data must have nonzero values for the default objective")

        sklearn_validation.check_array(self.low_frequency_data, 
                                       accept_sparse = False, 
                                       dtype = "numeric", 
                                       force_all_finite=True, 
                                       ensure_2d=False, 
                                       allow_nd=False);

        self.low_frequency_data = np.asarray(self.low_frequency_data)

        if (self.low_frequency_data.ndim != 1): 
            raise IndexError("array should be one dimensional")

        len_low = self.low_frequency_data.shape[0]
        len_high = self.high_frequency_data.shape[0]

        if (len_high % len_low != 0):
            raise IndexError("length of high frequency data must be divisible by length of low frequency data")
        else:
            self.frequency = int(len_high / len_low)

        # annotations 
        if self.annotation_data is None:
            self.annotation_data = np.full(self.high_frequency_data.shape, np.NaN)
            self.quadratic_constraint_weight = 0
        else:
            assert self.quadratic_constraint_weight is not None, "specify a value of the hyperparameter `quadradtic_constraint_weight`" 

        sklearn_validation.check_array(self.annotation_data, 
                                       accept_sparse = False, 
                                       dtype = "numeric", 
                                       force_all_finite=False, 
                                       ensure_2d=False, 
                                       allow_nd=False);

        self.annotation_data = np.asarray(self.annotation_data)

        if (self.annotation_data.ndim != 1): 
            raise IndexError("Array should be one dimensional")

        len_annotation = self.annotation_data.shape[0] 
        assert len_high == len_annotation, "length of annotation data must match length of high frequency data"

        # linear_constraint 
        if self.linear_constraint is None:
            self.linear_constraint = np.full(self.frequency, 1 / self.frequency)

        sklearn_validation.check_array(self.linear_constraint, 
                                       accept_sparse = False, 
                                       dtype = "numeric", 
                                       force_all_finite=True, 
                                       ensure_2d=False, 
                                       allow_nd=False);

        self.linear_constraint = np.asarray(self.linear_constraint)

        assert len(self.linear_constraint) == self.frequency, "length of linear constraint must equal length of high frequency data divided by length of low frequency data"

        # quadratic_constraint 
        if self.quadratic_constraint_weight is not None:
            sklearn_validation.check_scalar(self.quadratic_constraint_weight, "quadratic_constraint_weight", (float, int), min_val = 0)

    def _solver(self) -> ArrayLike:
        t = len(self.low_frequency_data)
        k = self.frequency
        T = len(self.high_frequency_data)

        constraints_matrix = np.kron(np.eye(t), self.linear_constraint.reshape(-1,1))

        inverse_high_frequency_data_matrix = np.diag(1.0 / self.high_frequency_data)
        cost_derivative_matrix = np.eye(T)

        diag_idx0, diag_idx1 = np.diag_indices(T)
        cost_derivative_matrix[diag_idx0[1:-1], diag_idx1[1:-1]] += 1
        cost_derivative_matrix[diag_idx0[:-1]+1, diag_idx1[:-1]] = -1
        cost_derivative_matrix[diag_idx0[:-1], diag_idx1[:-1]+1] = -1

        cost_matrix = 2 * np.dot(np.dot(inverse_high_frequency_data_matrix, cost_derivative_matrix), inverse_high_frequency_data_matrix)

        annotation_matrix = np.zeros(self.annotation_data.shape)
        annotation_matrix[~np.isnan(self.annotation_data)] = 1
        annotation_matrix = np.diag(annotation_matrix)
        annotation_matrix = 2 * self.quadratic_constraint_weight * annotation_matrix

        matrix = np.zeros((T+t, T+t))  
        matrix[:T,:T] = cost_matrix + annotation_matrix
        matrix[:T,T:] = constraints_matrix
        matrix[T:,:T] = constraints_matrix.T

        values = np.zeros((t+T,1)) 
        values[:T] = 2 * self.quadratic_constraint_weight * np.nan_to_num(self.annotation_data).reshape(-1,1)
        values[-t:] = self.low_frequency_data.reshape(-1,1) 

        X = np.linalg.solve(matrix, values)
        X = X[:-t] 

        return X.squeeze()