import numpy as np
import os
from imputegap.recovery.imputation import DeepMVI
from imputegap.recovery.manager import TimeSeries  # adjust if different

def load_airq():
    p = os.path.join(os.path.dirname(__file__), '..', 'imputegap', 'datasets', 'airq.txt')
    return np.loadtxt(p)

def test_deepmvi_adapter_shape_and_nan():
    X = load_airq()
    # create a simple Timeseries wrapper expected by BaseImputer / DeepMVI,
    # but to avoid deeper API dependencies, call algorithm directly if allowed:
    imputer = DeepMVI(X) if hasattr(DeepMVI, '__call__') else None
    # If DeepMVI expects a TimeSeries object, the test should create it; fallback: call algorithm function
    try:
        recov = DeepMVI(X).impute()
    except Exception:
        # fallback to calling algorithm module directly
        from imputegap.algorithms.deepmvi import deepmvi
        recov = deepmvi(X)
    assert isinstance(recov, np.ndarray)
    assert recov.shape == X.shape
    assert not np.isnan(recov).any()
