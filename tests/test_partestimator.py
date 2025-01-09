import pytest
import numpy as np
from partbte.partition import PartEstimator

# Helper function to generate mock data
def generate_mock_data(n_samples=100):
    y = np.random.choice(['A', 'B','C'], size=n_samples)
    treatment = np.random.choice([0, 1, 2], size=n_samples)
    y1_name = 'A'
    y0_name = 'C'
    treated_arms = [1,2]
    control_arms = [0]
    y_level = ['A', 'C']
    return y, treatment, y1_name, y0_name, treated_arms, control_arms, y_level

def test_partestimator_initialization():
    y, treatment, y1_name, y0_name, treated_arms, control_arms, y_level = generate_mock_data()
    estimator = PartEstimator(y, treatment, y1_name, y0_name, treated_arms, control_arms, y_level)
    
    assert isinstance(estimator, PartEstimator)
    assert hasattr(estimator, 'y1')
    assert hasattr(estimator, 'y0')

def test_fit_and_bounds():
    y, treatment, y1_name, y0_name, treated_arms, control_arms, y_level = generate_mock_data()
    estimator = PartEstimator(y, treatment, y1_name, y0_name, treated_arms, control_arms, y_level)
    
    pidx = np.random.choice([0, 1, 2, 3], size=np.sum(np.isin(y, y_level)))
    estimator.fit(pidx)
    
    bounds = estimator.pb_bounds()
    assert len(bounds) == 2
    assert bounds[0] <= bounds[1]  # Lower bound <= Upper bound

def test_naive_bounds():
    y, treatment, y1_name, y0_name, treated_arms, control_arms, y_level = generate_mock_data()
    estimator = PartEstimator(y, treatment, y1_name, y0_name, treated_arms, control_arms, y_level)
    
    naive_bounds = estimator.naive_bounds()
    assert len(naive_bounds) == 2
    assert naive_bounds[0] <= naive_bounds[1]
    
def test_simu_dist_and_ci():
    y, treatment, y1_name, y0_name, treated_arms, control_arms, y_level = generate_mock_data()
    estimator = PartEstimator(y, treatment, y1_name, y0_name, treated_arms, control_arms, y_level)
    
    # Fit a simple partition and simulate distributions
    pidx = np.random.choice([0, 1, 2, 3], size=np.sum(np.isin(y, y_level)))
    estimator.fit(pidx)
    estimator.simu_dist(nobs=1000)
    
    # Compute confidence intervals
    alpha = 0.1
    upp_ci = estimator.upp_ci(alpha=alpha)
    low_ci = estimator.low_ci(alpha=alpha)
    ext_ci = estimator.extended_ci(alpha=alpha)
    
    # Check the simulated confidence intervals
    assert upp_ci[1] <= 1  
    assert low_ci[0] >= 0  
    assert ext_ci[1] == upp_ci[1]
    assert ext_ci[0] == low_ci[0]
