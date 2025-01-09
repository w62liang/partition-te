import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from partbte.partition import PartLearner

# Helper function to generate mock data
def generate_mock_data(n_samples=200, n_features=5):
    X = np.random.rand(n_samples, n_features)
    y = np.random.choice(['A', 'B','C'], size=n_samples)
    treatment = np.random.choice([0, 1, 2], size=n_samples)
    return X, y, treatment

def test_partlearner_initialization():
    treated_clf = RandomForestClassifier()
    control_clf = RandomForestClassifier()
    y_level = ['A', 'C']
    treated_arms = [1, 2]
    control_arms = 0

    learner = PartLearner(
        y_level=y_level, treated_arms = treated_arms, 
        control_arms = control_arms, treated_clf=treated_clf, 
        control_clf=control_clf
    )
    assert isinstance(learner, PartLearner)

def test_fit():
    X, y, treatment = generate_mock_data()
    treated_clf = RandomForestClassifier()
    control_clf = RandomForestClassifier()
    y_level = ['A', 'C']
    treated_arms = [1, 2]
    control_arms = 0

    learner = PartLearner(
        y_level=y_level, treated_arms = treated_arms, 
        control_arms = control_arms, treated_clf=treated_clf, 
        control_clf=control_clf
    )
    learner.fit(X, y, treatment)

    assert hasattr(learner, 'treated_clf')
    assert hasattr(learner, 'control_clf')

def test_partition_plugin():
    X, y, treatment = generate_mock_data()
    treated_clf = RandomForestClassifier()
    control_clf = RandomForestClassifier()
    y_level = ['A','B']
    treated_arms = [1, 2]
    control_arms = 0

    learner = PartLearner(
        y_level=y_level, treated_arms = treated_arms, 
        control_arms = control_arms, treated_clf=treated_clf, 
        control_clf=control_clf
    )
    learner.fit(X, y, treatment)
    pidx = learner.partition_plugin(X, y, treatment)
    
    assert len(pidx) == np.sum(np.isin(y, y_level))

def test_point_estimate_impute():
    X, y, treatment = generate_mock_data()
    treated_clf = RandomForestClassifier()
    control_clf = RandomForestClassifier()
    y_level = ['A','B']
    treated_arms = [1, 2]
    control_arms = 0

    learner = PartLearner(
        y_level=y_level, treated_arms = treated_arms, 
        control_arms = control_arms, treated_clf=treated_clf, 
        control_clf=control_clf
    )
    point_estimate = learner.point_estimate_impute(X, y, treatment)
    assert 0 <= point_estimate <= 1
    
def test_cf_intv_estimate():
    """
    Test the cf_intv_estimate method of PartLearner.
    """
    # Generate mock data
    X, y, treatment = generate_mock_data(n_samples=200, n_features=5)
    
    # Initialize the PartLearner
    treated_clf = RandomForestClassifier(n_estimators=10, random_state=0)
    control_clf = RandomForestClassifier(n_estimators=10, random_state=0)
    y_level = ['A','B']
    treated_arms = [1, 2]
    control_arms = 0
    
    learner = PartLearner(
        y_level=y_level, treated_arms = treated_arms, 
        control_arms = control_arms, treated_clf=treated_clf, 
        control_clf=control_clf
    )
    
    # Run the cf_intv_estimate method
    results = learner.cf_intv_estimate(
        X, y, treatment,
        y1_name='A', y0_name='A',
        cv_fold=5,
        random_state=42,
        calib_method='isotonic',
        alpha=0.1,
        n_jobs=1
    )
    
    # Assert the results are as expected
    assert isinstance(results, dict)  # Ensure the output is a dictionary
    assert 'point_estimate' in results
    assert 'bounds' in results
    assert 'upper_ci' in results
    assert 'lower_ci' in results
    assert 'extended_ci' in results

    # Check the shapes and ranges of the outputs
    assert isinstance(results['point_estimate'], float)
    assert 0 <= results['point_estimate'] <= 1

    assert len(results['bounds']) == 2
    assert results['bounds'][0] <= results['bounds'][1]

    assert len(results['upper_ci']) == 2
    assert results['upper_ci'][0] <= results['upper_ci'][1]

    assert len(results['lower_ci']) == 2
    assert results['lower_ci'][0] <= results['lower_ci'][1]

    assert len(results['extended_ci']) == 2
    assert results['extended_ci'][0] <= results['extended_ci'][1]

