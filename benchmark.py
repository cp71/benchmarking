from typing import List, Dict, Callable, Union, Optional

import numpy as np
from numpy.typing import ArrayLike
np.set_printoptions(precision = 4)

from sklearn.cross_decomposition import CCA
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.utils.validation as sklearn_validation
from sklearn.model_selection import GridSearchCV


