"""
Test all optimizers
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from src.optimizer_runner import OptimizationRunner
    print("✓ OptimizationRunner imported")
except Exception as e:
    print(f"✗ OptimizationRunner import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.optimizers import CMAESOptimizer, TuRBOOptimizer
    print("✓ CMAESOptimizer and TuRBOOptimizer imported")
except Exception as e:
    print(f"✗ Optimizers import failed: {e}")

try:
    from src.bayesian_opt import BasketOptimizer
    print("✓ BasketOptimizer imported")
except Exception as e:
    print(f"✗ BasketOptimizer import failed: {e}")

print("\nAll import tests completed!")