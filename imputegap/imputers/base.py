"""
Minimal BaseImputer expected by ImputeGAP wrappers and tests.

This class intentionally minimal:
 - stores the input (incomp_data)
 - provides recov_data attribute to hold imputed output
 - defines an abstract impute() to override
 - provides a simple helper to call and return imputed numpy array when needed
"""
from typing import Any
import numpy as np

class BaseImputer:
    def __init__(self, incomp_data: Any, *args, **kwargs):
        """
        incomp_data: the ImputeGAP TimeSeries-like object (could be any).
        Subclasses should set self.recov_data (numpy.ndarray) in impute().
        """
        self.incomp_data = incomp_data
        self.recov_data = None
        self.algorithm = "baseimputer"
        self.params = kwargs.get("params", {})

    def impute(self, params: dict = None):
        """
        Run imputation. Subclasses must override this method and set self.recov_data.
        Return the imputed numpy.ndarray.
        """
        raise NotImplementedError("Subclasses must implement impute()")

    def fit_transform(self, params: dict = None):
        """
        Convenience wrapper: call impute and return the recovered matrix.
        """
        out = self.impute(params=params)
        # ensure recov_data is a numpy array
        if isinstance(out, np.ndarray):
            self.recov_data = out
        return self.recov_data
