"""
Optimizers module for basket trading weight optimization
"""
from .base_optimizer import BaseOptimizer
from .cma_es_optimizer import CMAESOptimizer
from .turbo_optimizer import TuRBOOptimizer
from .cvfs_cma_es_optimizer import CVFS_CMAESOptimizer

# Try to import enhanced optimizers
try:
    from .turbo_optimizer_tuned import TuRBOTunedOptimizer
except ImportError:
    class TuRBOTunedOptimizer:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("TuRBOTunedOptimizer not available")

try:
    from .saasbo_optimizer import SAASBOOptimizer
except ImportError:
    class SAASBOOptimizer:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("SAASBOOptimizer not available")

__all__ = [
    'BaseOptimizer', 
    'CMAESOptimizer', 
    'TuRBOOptimizer', 
    'CVFS_CMAESOptimizer',
    'TuRBOTunedOptimizer',
    'SAASBOOptimizer'
]