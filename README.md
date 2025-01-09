# Partition-TE
Partition-TE (`partbte`) is a Python package for statistical inference of treatment effect heterogeneity in randomized experiments. It is rooted in the philosophy of partitioning, and employs modern machine learning classification algorithms to provide rough point estimates and accurate interval estimates of heterogeneous treatment effects under binary outcomes such as treatment harm rate (THR) and treatment benefit rate (TBR).

The partitioning-based interval estimates provided by `partbte` are justified by randomization, meaning their validity (cover the true value with high probability) does not rely on any involved models or algorithms. Therefore, the interval estimates provided by `partbte` are highly reliable. However, whether the interval estimates provided are informative does rely on the underlying models. 

The `partbte` offers two key classes—`partition.PartEstimator` and `partition.PartLearner`. The `partition.PartEstimator` provides naive Frechet-Hoeffding bounds on the THR or TBR. It also allows estimation based on the user-specified partition. The `partition.PartLearner` interacts with `sklearn` and utilizes its estimators with the class `predict_proba` to realize the partitioning algorithm.

## Quick Start
We give an example on computing naive and partitionng-based bounds for the treatment harm rate (THR).
```python
import numpy as np

#%% define functions to generate data: covariates and potential outcomes.
def sigmoid(x):
    
    return 1 / (1 + np.exp(-x))

def generate_data(n_samples):
    
    X = np.random.normal(size = n_samples)
    py0 = sigmoid(X)
    py1 = sigmoid(X+1)
    py2 = sigmoid(X+2)

    y0 = np.where(
        np.random.uniform(0, 1, n_samples) < py0,  'alive',  'dead'
    )

    y1 = np.where(
        np.random.uniform(0, 1, n_samples) < py1,  'alive',  'dead'
    )

    y2 = np.where(
        np.random.uniform(0, 1, n_samples) < py2,  'alive',  'dead'
    )
    return X, y0, y1, y2
```
```python
#%% generate data
n_samples = 500
X, y0, y1, y2 = generate_data(n_samples)
treatment = np.random.choice(['med1','med2', 'placebo'], size=n_samples)
y = np.empty(n_samples, dtype=object)
y[treatment == 'placebo'] = y0[treatment == 'placebo']
y[treatment == 'med1'] = y1[treatment == 'med1']
y[treatment == 'med2'] = y2[treatment == 'med2']

# display data by pd.DataFrame
import pandas as pd
df = pd.DataFrame({'X': X, 'Y': y, 'Treatment': treatment})
print(df.head(10))
```
```
# Output:
          X      Y Treatment
0 -0.603908  alive      med2
1 -0.044319  alive      med1
2  0.565727   dead   placebo
3 -1.169742  alive      med2
4  0.115002  alive   placebo
5 -1.044254  alive      med2
6  0.541295  alive      med1
7 -0.589753   dead      med1
8 -0.680295   dead   placebo
9 -0.886229  alive      med1
```
```python
#%% define quantity of interest
treated_arms, control_arms = ['med1'], ['placebo'] # treated group: medication 1; control group: placebo
y0_name, y1_name = "alive", "dead" # pre-treatment outcome: alive; post-treatment outcome: dead. This yields the treatment harm rate (P(Y(0)=alive, Y(1)=dead)) of medication 1 (versus placebo). 
y_level = [y0_name, y1_name] # outcomes of interest
```
```python
#%% compute the naive bounds of the THR by PartEstimator
from partbte.partition import PartEstimator
naive_estimator = PartEstimator(
    y, treatment, y1_name, y0_name, treated_arms, control_arms, y_level)
alpha = 0.1 # significance level
print(f'Naive bounds for THR: {np.round(naive_estimator.naive_bounds(),3)}')

# evaluate the precisions of the estimated bounds
naive_estimator.simu_dist() # simulate the sampling distributions of the estimated bounds
print(f'{(1-alpha)*100}% confidence interval for the lower bound:', np.round(naive_estimator.low_ci(),3))
print(f'{(1-alpha)*100}% confidence interval for the upper bound:', np.round(naive_estimator.upp_ci(),3))
print(f'{(1-alpha)*100}% extended confidence interval for THR:', np.round(naive_estimator.extended_ci(),3))
```
```
# Output:
Naive bounds for THR: [0.    0.305]
90.0% confidence interval for the lower bound: [0. 0.]
90.0% confidence interval for the upper bound: [0.248 0.362]
90.0% extended confidence interval for THR: [0.    0.362]
# The output implies the THR lies in the interval [0, 0.305] with high probability.
```
```python
#%% compute the bounds of THR under user-specified stratification
pidx = (X>-1)[np.isin(treatment, np.hstack([treated_arms, control_arms]))] # stratify the data by X+1<=0 and X+1>0. The length of pidx must match with the observations in the treatment gruops.
naive_estimator.fit(pidx) # fit the partition 
print(f'Post-stratification bounds for THR: {np.round(naive_estimator.pb_bounds(),3)}')

# evaluate the precisions of the estimated bounds
naive_estimator.simu_dist() # simulate the sampling distributions of the estimated bounds
print(f'{(1-alpha)*100}% confidence interval for the lower bound:', np.round(naive_estimator.low_ci(),3))
print(f'{(1-alpha)*100}% confidence interval for the upper bound:', np.round(naive_estimator.upp_ci(),3))
print(f'{(1-alpha)*100}% extended confidence interval for THR:', np.round(naive_estimator.extended_ci(),3))
```
```
# Output
User-specified post-stratification bounds for THR: [0.    0.252]
90.0% confidence interval for the lower bound: [0. 0.]
90.0% confidence interval for the upper bound: [0.199 0.304]
90.0% extended confidence interval for THR: [0.    0.304]
```
```python
#%% partitioning-based bounds for THR
from sklearn.linear_model import LogisticRegression
from partbte.partition import PartLearner
maBounds = PartLearner( # construct a base learner
    treated_clf = LogisticRegression(), control_clf = LogisticRegression(), # specify the models for the treated and control groups
    treated_arms = treated_arms, control_arms = control_arms,
    y_level = ['alive','dead'])

bounds_est = maBounds.cf_intv_estimate( # estimate the THR by cross-fitting partitioning-based algorithms
    X = X, y = y, treatment = treatment, y1_name = y1_name, y0_name = y0_name, 
    cv_fold = 2, random_state = 0, calib_method = None,  
    alpha = alpha
)

print('A rough point estimate of THR by imputation: ', np.round(bounds_est['point_estimate'],3))
print('Partitioning-based bounds of THR: ', np.round(bounds_est['bounds'],3))
print(f'{(1-alpha)*100}% confidence interval for the lower bound:', np.round(bounds_est['lower_ci'],3))
print(f'{(1-alpha)*100}% confidence interval for the upper bound:', np.round(bounds_est['upper_ci'],3))
print(f'{(1-alpha)*100}% extended confidence interval for THR:', np.round(bounds_est['extended_ci'],3))
```
```
# Output
Cross-Fitting Progress: 100%|██████████| 2/2 [00:00<00:00, 19.00it/s]
A rough point estimate of THR by imputation:  0.037
Partitioning-based bounds of THR:  [0.    0.199]
90.0% confidence interval for the lower bound: [0. 0.]
90.0% confidence interval for the upper bound: [0.128 0.27 ]
90.0% extended confidence interval for THR: [0.   0.27]
```
```python
#%% Consider other classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

classifiers = {
    "Lgit": LogisticRegression(),
    "NBayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RF": RandomForestClassifier(max_depth=5, n_estimators = 200),
}

method = list(classifiers.keys())
from collections import defaultdict
output_dict = {c_key: defaultdict(list) for c_key in method}

# 
from partbte.partition import PartLearner
for k,v in classifiers.items():
    t_clf, c_clf = v.__class__(**v.get_params()), v.__class__(**v.get_params())
    maBounds = PartLearner( # construct a base learner
        treated_clf = t_clf, control_clf = c_clf, 
        treated_arms = treated_arms, control_arms = control_arms,
        y_level = ['alive','dead'])

    intv_est = maBounds.cf_intv_estimate( # estimate the THR by cross-fitting partitioning-based algorithms
        X = X, y = y, treatment = treatment, y1_name = y1_name, y0_name = y0_name, 
        cv_fold = 2, random_state = 0, calib_method = None,  
        alpha = alpha
    )
    
    output_dict[k] = intv_est

# round the values to three decimal places.
def round_element(val, decimal = 3):
    
    if decimal == None:
        return val
    elif isinstance(val, list): 
        return [round(x, decimal) for x in val]
    elif isinstance(val, np.ndarray):  
        return np.round(val, decimal)
    elif isinstance(val, (int, float)): 
        return round(val, decimal)
    else:
        return val 
    
# paritioning-based interval estimates
import pandas as pd
df = pd.DataFrame.from_dict(output_dict, orient='index')
for i in df.columns:
    df[i] = df[i].map(lambda x: round_element(x, decimal = 3))

pd.set_option('display.max_columns', None)
print(df)
```
```
# Output
Cross-Fitting Progress: 100%|██████████| 2/2 [00:00<00:00, 19.14it/s]
Cross-Fitting Progress: 100%|██████████| 2/2 [00:00<00:00, 25.45it/s]
Cross-Fitting Progress: 100%|██████████| 2/2 [00:00<00:00, 23.42it/s]
Cross-Fitting Progress: 100%|██████████| 2/2 [00:00<00:00, 13.92it/s]
Cross-Fitting Progress: 100%|██████████| 2/2 [00:00<00:00,  2.40it/s]
        point_estimate        bounds        upper_ci    lower_ci   extended_ci
Lgit             0.037  [0.0, 0.199]   [0.128, 0.27]  [0.0, 0.0]   [0.0, 0.27]
NBayes           0.037  [0.0, 0.196]  [0.124, 0.267]  [0.0, 0.0]  [0.0, 0.267]
SVM              0.031  [0.0, 0.243]   [0.169, 0.32]  [0.0, 0.0]   [0.0, 0.32]
KNN              0.077  [0.0, 0.217]  [0.148, 0.293]  [0.0, 0.0]  [0.0, 0.293]
RF               0.043   [0.0, 0.22]  [0.156, 0.301]  [0.0, 0.0]  [0.0, 0.301]
```
