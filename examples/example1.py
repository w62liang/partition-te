import numpy as np

#%% define functions for data generation
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

#%% generate data
n_samples = 500
X, y0, y1, y2 = generate_data(n_samples)
treatment = np.random.choice(['med1','med2', 'placebo'], size=n_samples)
y = np.empty(n_samples, dtype=object)
y[treatment == 'placebo'] = y0[treatment == 'placebo']
y[treatment == 'med1'] = y1[treatment == 'med1']
y[treatment == 'med2'] = y2[treatment == 'med2']

import pandas as pd
df = pd.DataFrame({'X': X, 'Y': y, 'Treatment': treatment})
print(df.head(10))

#%% define quantity of interest
treated_arms, control_arms = ['med1'], ['placebo'] # treated group: medication 1; control group: placebo
y0_name, y1_name = "alive", "dead" # pre-treatment outcome: alive; post-treatment outcome: dead. This yields the treatment harm (THR) rate of medication 1 versus placebo
y_level = [y0_name, y1_name] # outcomes of interest

#%% compute the naive bounds of the THR
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


#%% compute the bounds of THR under stratification
pidx = (X>-1)[np.isin(treatment, np.hstack([treated_arms, control_arms]))] # stratify the data by X+1<=0 and X+1>0. The length of pidx must match with the observations in the treatment gruops.
naive_estimator.fit(pidx) # fit the partition 
print(f'Post-stratification bounds for THR: {np.round(naive_estimator.pb_bounds(),3)}')

# evaluate the precisions of the estimated bounds
naive_estimator.simu_dist() # simulate the sampling distributions of the estimated bounds
print(f'{(1-alpha)*100}% confidence interval for the lower bound:', np.round(naive_estimator.low_ci(),3))
print(f'{(1-alpha)*100}% confidence interval for the upper bound:', np.round(naive_estimator.upp_ci(),3))
print(f'{(1-alpha)*100}% extended confidence interval for THR:', np.round(naive_estimator.extended_ci(),3))

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