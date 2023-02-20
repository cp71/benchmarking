from typing import List, Dict, Callable, Union, Optional

import numpy as np
from numpy.typing import ArrayLike

from sklearn.cross_decomposition import CCA
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.utils.validation as sklearn_validation

from . import solvers 

class Estimator(BaseEstimator, TransformerMixin):
    """ 
    A scikit-learn compatible estimator for benchmarking.
    
    Parameters
    ----------
    linear_constraint: Optional[ArrayLike] = None
        linear constraint in optimization problem. 
        default is average
        
    quadratic_constraint_weight: Optional[Union[float,int]] = None
        weight of quadratic constraint in relaxation of optimization problem
        default is 0
    
    solver: Optional[Callable] = None
        solver for custom objective in optimiation
        default is consecutive differences of ratio between up-sampled data and high-frequency data
    
    Methods
    -------
    fit: Callable 
        determine optimization problem with high frequency data and low frequency data along with annotations
    
    transform: Callable 
        projection of high frequency data through Canonical Correlation Analysis with low frequency data
    
    predict: Callable
        solution of optimization problem 
    
    score: Callable
        correlation between low frequency data and up-sampled data in Canonical Correlation Analysis
    
    Attributes
    ----------
    linear_constraint: ArrayLike
        linear constraint in optimization problem
        
    quadratic_constraint_weight: Union[float,int]]
        weight of quadratic constraint in relaxation of optimization problem
    
    solver: Callable
        solver for custom objective in optimiation
    
    high_frequency_data: ArrayLike
        high frequency dataset
        
    low_frequency_data: ArrayLike
        low frequency dataset
        
    annotations: ArrayLike
        predictions of subject matter expert

    interpolant: ArrayLike
        up-sampled dataset from solution to optimization problem

    projection: ArrayLike
        projects from `transform` method
    
    frequency: int
        Length of high_frequency_data divided by length of low_frequency_data
    
    
    Example
    -------
    >>> high_frequency_data = np.array([50,100,150,100] * 5)
    >>> low_frequency_data = np.array([500,400,300,400,500])
    >>> estimator = benchmark.Estimator()
    >>> estimator.fit(high_frequency_data, low_frequency_data)
    >>> estimator.predict()
    array([257.3392, 511.2246, 751.2952, 480.141 , 226.2556, 423.9027,590.0058, 359.8359, 162.1888, 297.7839, 433.3789, 306.6484,171.0534, 376.5866, 613.6638, 438.6962, 233.163 , 490.5022, 761.6564, 514.6784])
    
    Notes
    -----
    For more information, please consult the :ref:`User Guide <user_guide>`.
    
    See Also
    --------
    statsmodels.tsa.interp.denton
    """
    
    
    def __init__(self, linear_constraint: Optional[ArrayLike] = None, quadratic_constraint_weight: Optional[Union[float,int]] = None, solver: Optional[Callable] = None):
        self.linear_constraint = linear_constraint
        self.quadratic_constraint_weight = quadratic_constraint_weight
        self.solver = solver
         
    def fit(self, high_frequency_data: ArrayLike, low_frequency_data: ArrayLike, annotation_data: Optional[ArrayLike] = None):
        """
        fit estimator to high frequency data and low frequency data along with annotations
        
        Parameters
        ----------
        high_frequency_data: ArrayLike
            high frequency dataset
        
        low_frequency_data: ArrayLike
            low frequency dataset
        
        annotation_data: Optional[ArrayLike] = None
            predictions of subject matter expert
            
        Example
        -------
        >>> high_frequency_data = np.array([50,100,150,100] * 5)
        >>> low_frequency_data = np.array([500,400,300,400,500])
        >>> estimator = benchmark.Estimator()
        >>> estimator.fit(high_frequency_data, low_frequency_data)
        """
        self.high_frequency_data = high_frequency_data
        self.low_frequency_data = low_frequency_data
        self.annotation_data = annotation_data
        
        self._validate()
        
        if self.solver is None:
            self.interpolant = solvers.exact(self.high_frequency_data, 
                                             self.low_frequency_data, 
                                             self.annotation_data, 
                                             self.linear_constraint, 
                                             self.quadratic_constraint_weight) 
        else:
            self.interpolant = self.solver(self.high_frequency_data, self.low_frequency_data, self.annotation_data, self.linear_constraint, self.quadratic_constraint_weight) 
        
        return self

    def transform(self, X=None):
        """
        project high frequency data to low frequency data through Canonical Correlation Analysis
            
        Example
        -------
        >>> high_frequency_data = np.array([50,100,150,100] * 5)
        >>> low_frequency_data = np.array([500,400,300,400,500])
        >>> estimator = benchmark.Estimator()
        >>> estimator.fit(high_frequency_data, low_frequency_data)
        >>> estimator.transform()
        
        See Also
        --------
        sklearn.cross_decomposition.CCA
        """
        
        model = CCA(n_components=1)
        X_c, Y_c = model.fit_transform(self.high_frequency_data.reshape(-1,self.frequency), self.low_frequency_data.reshape(-1,1))

        self.projection = model.x_rotations_.flatten()
        return np.squeeze(X_c)
    
    def predict(self, X = None):
        """
        return up-sampled data from solution to optimization problem
        """
        sklearn_validation.check_is_fitted(self, 'interpolant')
        return self.interpolant
    
    def score(self, X = None, y = None):
        """
        calculate correlation between up-sampled data and low frequency data through Canonical Correlation Analysis

        See Also
        --------
        sklearn.cross_decomposition.CCA
        """
        
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