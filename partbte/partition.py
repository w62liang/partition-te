import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
from joblib import Parallel, delayed
from .utils import convert_to_np, filter_data, safe_div, Minrow_idx
from scipy.stats import multivariate_hypergeom

#%%
class PartEstimator:
    """
    A class for statistical inference of treatment effect heterogeneity under
    randomization.

    This class computes bounds on the proportitions of individuals who responde
    differently to the treatment, e.g., the treatment harm rate and treatment 
    benefit rate, based on a provided partition. It also computes confidence
    intervals for the estimated bounds, and provide an extended confidence
    interval for the proportion of interest after Bonferroni adjustment. If 
    needed, a point estimator of the rate of interest can be given under the
    conditional independence assumption for the potential outcomes.
    
    The true value of the rate will be within or at least on the boundary of 
    the interval estimator. This holds with probability tending to 1, without 
    requiring any identification or model assumptions but simple randomization,
    e.g., the completely randomized experiment or the Bernoulli trials. 
    
    Args:
        y (array-like, shape (n_samples,)): A vector of observed outcomes 
            where n_samples is the sample size.
        treatment (array-like, shape (n_samples,)): Treatment group labels.
        y1_name (str or float): Post-treatment outcome of interest. A one-dim
            value.
        y0_name (str or float): Pre-treatment outcome of interest. A one-dim
            value.
        treated_arms (int, str, list or np.ndarray): Arms (or groups) defined 
            as the reated group.
        control_arms (int, str, list or np.ndarray): Arms (or groups) defined 
            as the control group.
        y_level (list or np.ndarray): Outcome values of interest.
        
    Example::
        
        
    """
    
    def __init__(
            self,
            y,
            treatment,
            y1_name,
            y0_name,
            treated_arms,
            control_arms,
            y_level,
            ):
        
        params = {'y':y, 'treatment': treatment, 'treated_arms': treated_arms,
                  'control_arms': control_arms, 'y_level': y_level}
        a, y = filter_data(**params)
        self.y1 = np.isin(y, y1_name)
        self.y0 = np.isin(y, y0_name)
        self.ispidx = True
        self.a = a
    
    def fit(self, pidx):
        """
        Fits given partition indices.

        Args:
            pidx (np.ndarray): Partition indices for data.

        Updates:
            musd (list): Contains mean, standard deviation, partition weights 
                and bounds for each partition.
        """
        if len(pidx)!=len(self.a):
            raise ValueError(
                "The pidx must match with the number of your observations in\
                y_level and treatment arms."
                )
        plabel, counts = np.unique(pidx, return_counts=True)
        rho_values = counts / len(pidx)

        sub_masks = [pidx == label for label in plabel]
        a_subs = [self.a[mask] for mask in sub_masks]
        y1_subs = [self.y1[mask] for mask in sub_masks]
        y0_subs = [self.y0[mask] for mask in sub_masks]

        sum_t_subs = np.array([np.sum(a) for a in a_subs])
        sum_c_subs = np.array([len(a) - s for a, s in zip(a_subs, sum_t_subs)])
        
        rate_t = np.array([np.mean(a) for a in a_subs])
        rate_c = np.array([np.mean(1-a) for a in a_subs])

        avey1s = np.array([safe_div(np.sum(y1 * a), s) for y1, a, s in 
                           zip(y1_subs, a_subs, sum_t_subs)])
        avey0s = np.array([safe_div(np.sum(y0 * (1 - a)), s) for y0, a, s in 
                           zip(y0_subs, a_subs, sum_c_subs)])

        sdy1s = np.array([safe_div((avey1 - avey1**2), max(s - 1, 0))**0.5 for 
                          avey1, s in zip(avey1s, sum_t_subs)])
        sdy0s = np.array([safe_div((avey0 - avey0**2), max(s - 1, 0))**0.5 for 
                          avey0, s in zip(avey0s, sum_c_subs)])
        
        sigmay1s = (avey1s - avey1s**2)*sum_t_subs
        sigmay0s = (avey0s - avey0s**2)*sum_c_subs
        
        self.musd = {'mu1': avey1s, 'mu0': avey0s, 'sd1': sdy1s, 'sd0': sdy0s,
                     'sigma1': sigmay1s, 'sigma0': sigmay0s,
                     'rho': rho_values, 'rate_t': rate_t, 'rate_c': rate_c,
                     'low': np.maximum(0, avey0s + avey1s - 1),
                     'upp': np.minimum(avey0s, avey1s)}
        self.ispidx = False

    def pb_bounds(self):
        """
        Computes the partitioning-based bounds after fitting a partition.

        Returns:
            np.ndarray: Interval estimate as [lower_bound, upper_bound].
        """
        if self.ispidx:
            raise AttributeError(
                "You must fit a pidx to before running pb_bounds. Otherwise,\
                 run naive_bounds")
        lowb = np.sum(self.musd['low'] * self.musd['rho'])
        uppb = np.sum(self.musd['upp'] * self.musd['rho'])
        return np.array([lowb, uppb], dtype = "float")
    
    def naive_bounds(self):
        """
        Computes naive bounds assuming no partitioning.

        Returns:
            np.ndarray: Naive interval estimate for the treatment effect 
                heterogeneity.
        """
        pidx = np.ones(len(self.y1))
        self.fit(pidx)
        return self.pb_bounds()
    
    def simu_dist(self, nobs=50000):
        """
        Simulates distributions of the estimated bounds using Monte Carlo 
        replications.

        Args:
            nobs (int): Number of Monte Carlo replications. Default is 5*e4

        Updates:
            uppq (np.ndarray): Simulated upper bound estimates.
            lowq (np.ndarray): Simulated lower bound estimates.
        """
        if self.ispidx:
            pidx = np.ones(len(self.y1))
            self.fit(pidx)
        n = len(self.a)
        
        # # normal approximation
        # rho_prod = np.outer(self.musd['rho'], self.musd['rho'])
        # sigma1 = np.sum((self.musd['low'][:,None] 
        #                   - self.musd['low'][None:,])**2*rho_prod)/2/n
        # sigma2 = np.sum((self.musd['upp'][:,None] 
        #                   - self.musd['upp'][None:,])**2*rho_prod)/2/n

        # qsiL = np.random.normal(loc=0, scale=sigma1**0.5, size=nobs)
        # qsiU = np.random.normal(loc=0, scale=sigma2**0.5, size=nobs)
        
        # qmu1 = np.random.normal(loc=self.musd['mu1'][:,None], 
        #                         scale=self.musd['sd1'][:,None], 
        #                         size= (len(self.musd['mu1']), nobs))
        # qmu0 = np.random.normal(loc=self.musd['mu0'][:,None], 
        #                         scale=self.musd['sd0'][:,None], 
        #                         size= (len(self.musd['mu0']), nobs))
        
        # lowv = np.maximum(0, qmu1 + qmu0 - 1) * self.musd['rho'][:, None]
        # uppv = np.minimum(qmu1, qmu0) * self.musd['rho'][:, None]
        # self.uppq = np.sum(uppv, axis=0) + qsiU
        # self.lowq = np.sum(lowv, axis=0) + qsiL
        
        # # multivariate hypergeometric approximation        
        # rho_values = self.musd['rho']
        # p = len(rho_values)
        # colors = (rho_values*100000).astype('int')
        
        # q_n = multivariate_hypergeom.rvs(colors, n, size=nobs)/n
        
        # qmu1 = np.random.normal(loc=self.musd['mu1'][:,None], 
        #                         scale=self.musd['sd1'][:,None], 
        #                         size= (p, nobs))
        # qmu0 = np.random.normal(loc=self.musd['mu0'][:,None], 
        #                         scale=self.musd['sd0'][:,None], 
        #                         size= (p, nobs))
        
        # lowv = np.maximum(0, qmu1 + qmu0 - 1) * q_n.T
        # uppv = np.minimum(qmu1, qmu0) * q_n.T

        # self.uppq = np.sum(uppv, axis=0) 
        # self.lowq = np.sum(lowv, axis=0) 
        
        # conditional multivariate hypergeometric approximation
        rho_values = self.musd['rho']
        p = len(rho_values)
        colors = (rho_values*n*50).astype('int')
        q_n = multivariate_hypergeom.rvs(colors, n, size=nobs)
        mu1 = np.broadcast_to(self.musd['mu1'], (nobs, p))
        mu0 = np.broadcast_to(self.musd['mu0'], (nobs, p))
        qn_t = q_n*self.musd['rate_t']
        qn_c = q_n*self.musd['rate_c']
        scale1 = (self.musd['sigma1'] / np.maximum((qn_t-1)*qn_t,0.001))**0.5
        scale0 = (self.musd['sigma0'] / np.maximum((qn_c-1)*qn_c,0.001))**0.5
        
        qmu1 = np.random.normal(loc=mu1, scale=scale1, size= (nobs, p))
        qmu0 = np.random.normal(loc=mu0, scale=scale0, size= (nobs, p))
        
        lowv = np.maximum(0, qmu1 + qmu0 - 1) * q_n / n
        uppv = np.minimum(qmu1, qmu0) * q_n / n

        self.uppq = np.sum(uppv, axis=1) 
        self.lowq = np.sum(lowv, axis=1) 
    
    def _compute_ci(self, qv, alpha):
        """
        Helper function to compute confidence intervals.
    
        Args:
            qv (np.ndarray): Array of computed values.
            alpha (float): Significance level.
    
        Returns:
            np.ndarray: Confidence interval as [lower_limit, upper_limit].
        """
        qv = np.clip(qv, 0, 1)  # Ensure values are within [0, 1].
        qv_sorted = np.sort(qv)
        up = qv_sorted[::-1][int(alpha * len(qv) / 2)]
        low = qv_sorted[int(alpha * len(qv) / 2)]
        return np.array([low, up], dtype="float")

    def upp_ci(self, alpha=0.1):
        """
        Computes confidence interval for the upper bound.
    
        Args:
            alpha (float): Significance level.
    
        Returns:
            np.ndarray: Confidence intervals as [lower_limit, upper_limit].
        """
        uppb = np.sum(self.musd['upp'] * self.musd['rho'])
        qv = 2 * uppb - self.uppq
        return self._compute_ci(qv, alpha)
    
    def low_ci(self, alpha=0.1):
        """
        Computes confidence intervals for the lower bound.
    
        Args:
            alpha (float): Significance level.
    
        Returns:
            np.ndarray: Confidence interval as [lower_limit, upper_limit].
        """
        lowb = np.sum(self.musd['low'] * self.musd['rho'])
        qv = 2 * lowb - self.lowq
        return self._compute_ci(qv, alpha)
    
    def extended_ci(self, alpha=0.1):
        """
        Computes extended confidence interval combining both bounds.
    
        Args:
            alpha (float): Significance level.
    
        Returns:
            np.ndarray: Extended confidence interval for the proportition of
                interest.
        """
        uppb = np.sum(self.musd['upp'] * self.musd['rho'])
        lowb = np.sum(self.musd['low'] * self.musd['rho'])
        qv_upper = 2 * uppb - self.uppq
        qv_lower = 2 * lowb - self.lowq
    
        qv_upper_ci = self._compute_ci(qv_upper, alpha)
        qv_lower_ci = self._compute_ci(qv_lower, alpha)
    
        return np.array([qv_lower_ci[0], qv_upper_ci[1]], dtype="float")

#%%
class PartLearner:
    """
    A class for interval estimation of treatment effect heterogeneity and space
    partitioning by probabilistic classification algorithms.
    
    Args:
        treated_clf: (probabilistic classifier implementing 'fit' and 
            'predict_proba'): The object to fit the treated data.
        control_clf: (probabilistic classifier implementing 'fit' and 
            'predict_proba'): The object to fit the control data.
        treated_arms (int, str, list or np.ndarray): Arms (or groups) defined 
            as the reated group.
        control_arms (int, str, list or np.ndarray): Arms (or groups) defined 
            as the control group.        
        y_level (list or np.ndarray): Outcome values of interest.
    """

    def __init__(
        self,
        y_level,
        treated_clf, 
        control_clf,
        treated_arms,
        control_arms,
    ):
        if not hasattr(treated_clf, "fit") or not\
               hasattr(treated_clf, "predict_proba"):
            raise ValueError(
                "treated_clf must implement 'fit' and 'predict_proba' methods."
                )
        if not hasattr(control_clf, "fit") or not\
               hasattr(control_clf, "predict_proba"):
            raise ValueError(
                "control_clf must implement 'fit' and 'predict_proba' methods."
                )

        self.tclf_class = treated_clf.__class__
        self.cclf_class = control_clf.__class__
        self.tclf_params = treated_clf.get_params()
        self.cclf_params = control_clf.get_params()
        self.tbase = treated_clf
        self.cbase = control_clf
        self.treated_arms = treated_arms
        self.control_arms = control_arms
        self.y_level = y_level

    def fit(self, X, y, treatment):
        """
        Trains the treated and control classifiers using the provided data.

        Args:
            X (array-like): Feature matrix of shape (n_train, n_features).
            y (array-like): A vector of observed outcomes of shape 
                (n_train,).
            treatment (array-like): Treatment group labels of shape 
                (n_train,).
            
            The input data are used for training classifiers.

        Raises:
            ValueError: If the lengths of X, y, and treatment do not match.
        """
        # Ensure input lengths are consistent
        if len(X) != len(y) or len(X) != len(treatment):
            raise ValueError("X, y, and treatment must have the same length.")

        # Filter and split the data based on treatment groups
        x, a, y = filter_data(
            X=X, treatment=treatment, y=y, 
            y_level=self.y_level,
            treated_arms=self.treated_arms, 
            control_arms=self.control_arms
        )

        # Initialize classifiers for treated and control groups
        self.treated_clf = self.tclf_class(**self.tclf_params)
        self.control_clf = self.cclf_class(**self.cclf_params)
        
        # if x.ndim == 1:
        #     x = x.reshape(-1, 1)
        # Fit the classifiers to the respective data
        self.treated_clf.fit(x[a == 1], y[a == 1])
        self.control_clf.fit(x[a == 0], y[a == 0])

    def fit_calibcv(self, X, y, treatment, calib_method='isotonic'):
        """
        Trains the treated and control classifiers using the provided data and
            under model calibration.

        Args:
            X (array-like): Feature matrix of shape (n_train, n_features).
            y (array-like): A vector of observed outcomes of shape 
                (n_train,).
            treatment (array-like): Treatment group labels of shape 
                (n_train,).
            calib_method (str): Method to use for calibration. Two options:
                {'isotonic','sigmoid'}. Defaults to 'isotonic'.
                
            The input data are used for training classifiers.

        Returns:
            None.

        """
        # Ensure input lengths are consistent
        if len(X) != len(y) or len(X) != len(treatment):
            raise ValueError("X, y, and treatment must have the same length.")
        if calib_method not in ["isotonic","sigmoid"]:
            raise ValueError("The argument calib_method takes values in\
                             {'isotonic','sigmoid'}.")
        x, a, y = filter_data(
            X=X, treatment=treatment, y=y, 
            y_level=self.y_level,
            treated_arms=self.treated_arms, 
            control_arms=self.control_arms
        )
        self.treated_clf = CalibratedClassifierCV(self.tbase, 
                                                  method=calib_method)
        self.control_clf = CalibratedClassifierCV(self.cbase, 
                                                  method=calib_method)
        self.treated_clf.fit(x[a == 1], y[a == 1])
        self.control_clf.fit(x[a == 0], y[a == 0])

    def partition_plugin(self, X, y, treatment):
        """
        Creates partition indices by plugging in fitted classifiers.

        Args:
            X (array-like): Feature matrix of shape (n_estimate, n_features).
            y (array-like): A vector of observed outcomes of shape 
                (n_estimate,).
            treatment (array-like): Treatment group labels of shape 
                (n_estimate,).
                
            The input data are used for interval estimation.

        Returns:
            np.ndarray: partition indices of shape (n_samples, ).

        """
        x, a, y = filter_data(
            X=X, treatment=treatment, y=y, 
            y_level=self.y_level,
            treated_arms=self.treated_arms, 
            control_arms=self.control_arms
        )
        self.pred_dat = {'a': a, 'y': y}
        y1_prob = self.treated_clf.predict_proba(x)
        y0_prob = self.control_clf.predict_proba(x)
        return Minrow_idx(y1_prob, y0_prob)

    def point_estimate_impute(self, X, y, treatment, y1_name = 0, y0_name = 1):
        """
        Computes a point estimate by imputing the missing potential outcomes
        with predicted labels of the classifiers.

        Args:
            X (array-like): Feature matrix of shape (n_estimate, n_features).
            y (array-like): A vector of observed outcomes of shape 
                (n_estimate,).
            treatment (array-like): Treatment group labels of shape 
                (n_estimate,).
            y1_name (str or float): Post-treatment outcome of interest. A 
                one-dimensional value. Default is 0.
            y0_name (str or float): Pre-treatment outcome of interest. A 
                one-dimensional value. Default is 1.
                
            If y=1 is favored, then setting (y1_name=0, y0_name=1) yields the
                treatment harm rate.

        Returns:
            float: point estimate based on imputation.

        """
        x, a, y = filter_data(
            X=X, treatment=treatment, y=y, 
            y_level=self.y_level,
            treated_arms=self.treated_arms, 
            control_arms=self.control_arms
        )
        treated_clf = self.tclf_class(**self.tclf_params)
        control_clf = self.cclf_class(**self.cclf_params)
        treated_clf.fit(x[a == 1], y[a == 1])
        control_clf.fit(x[a == 0], y[a == 0])
        pred1, pred0 = treated_clf.predict(x), control_clf.predict(x)
        
        pred1[a == 1] = y[a == 1]
        pred0[a == 0] = y[a == 0]

        # Calculate the mean proportion where predictions match target labels
        matches_treated = (pred1 == y1_name)
        matches_control = (pred0 == y0_name)
        combined_matches = matches_treated & matches_control

        return np.mean(combined_matches)

    def _process_fold(
        self, X, y, treatment, y1_name, y0_name, calib_method, alpha, cv_idx, i
    ):
        pred_idx = cv_idx[i]
        fit_idx = np.concatenate([cv_idx[j] for j in range(len(cv_idx)) 
                                  if j != i])

        if calib_method is None:
            self.fit(X[fit_idx], y[fit_idx], treatment[fit_idx])
        else:
            self.fit_calibcv(X[fit_idx], y[fit_idx], treatment[fit_idx], 
                             calib_method)

        pidx = self.partition_plugin(X[pred_idx], y[pred_idx], 
                                     treatment[pred_idx])
        bounds = PartEstimator(
            y = y[pred_idx], treatment = treatment[pred_idx], 
            y1_name = y1_name, y0_name = y0_name, 
            treated_arms=self.treated_arms, 
            control_arms=self.control_arms, y_level = self.y_level,
        )
        bounds.fit(pidx)
        bounds.simu_dist()

        return bounds.pb_bounds(),bounds.upp_ci(alpha), bounds.low_ci(alpha),\
               bounds.extended_ci(alpha)
               

    def cf_intv_estimate(
        self, X, y, treatment, y1_name, y0_name, cv_fold=5, 
        random_state=0, calib_method=None, calibcv=5, alpha=0.5,
        n_jobs=None,
    ):
        """
        Estimates partitioning-based upper and lower bound of treatment effect 
        heterogeneity by cross-fitting. Computes confidence intervals for the
        estimated bounds and extended confidence interval for the proportion
        of interest.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): A vector of observed outcomes of shape 
                (n_samples,).
            treatment (array-like): Treatment group labels of shape 
                (n_samples,).
            y1_name (str or float): Post-treatment outcome of interest. A 
                one-dimensional value. 
            y0_name (str or float): Pre-treatment outcome of interest. A 
                one-dimensional value. 
            cv_fold (int, optional): Number of cross-validation folds. 
                Default is 5.
            random_state (int, optional): Seed for reproducibility. 
                Default is 0.
            calib_method (str, optional): Calibration method for classifiers 
                ({'isotonic', 'sigmoid'}). Default is None, i.e., no model
                calibration.
            calibcv (int, optional): Number of folds for calibration 
                cross-fitting. Default is 5.
            alpha (float, optional): Significance level for confidence 
                intervals. Default is 0.5.
            n_jobs (int, optional): Number of parallel jobs for computation. 
                Default is None.

        Returns:
            dict: A dictionary containing the estimated bounds and 
                confidence intervals.
            - 'bounds': Average bounds across folds.
            - 'upper_ci': Upper confidence interval.
            - 'lower_ci': Lower confidence interval.
            - 'extended_ci': Extended confidence interval.

        Raises:
        RuntimeError: If an error occurs during parallel execution.
        """
        assert (cv_fold >0) and (isinstance(cv_fold, int)),\
            "cv_fold must be a positive integer."
        np.random.seed(random_state)
        X, treatment, y = convert_to_np(X, treatment, y)
        n = len(y)
        sidx = np.arange(n)
        np.random.shuffle(sidx)
        cv_idx = np.array_split(sidx, cv_fold)

        # try:
        #     results = Parallel(n_jobs=n_jobs)(
        #         delayed(self._process_fold)(X, y, treatment, y1_name, y0_name, 
        #                                     calib_method, alpha, cv_idx, i)
        #         for i in tqdm(range(cv_fold), desc="Cross-Fitting Progress")
        #     )
        # except Exception as e:
        #     raise RuntimeError(f"Error during parallel execution: {str(e)}")
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_fold)(X, y, treatment, y1_name, y0_name, 
                                        calib_method, alpha, cv_idx, i)
            for i in tqdm(range(cv_fold), desc="Cross-Fitting Progress")
        )
        
        bounds_list, uppci_list, lowci_list, extci_list = zip(*results)

        return {
            'point_estimate': self.point_estimate_impute(
                X, y, treatment,y1_name, y0_name),
            'bounds': np.array(sum(bounds_list) / cv_fold, dtype='float'),
            'upper_ci': np.array(sum(uppci_list) / cv_fold, dtype='float'),
            'lower_ci': np.array(sum(lowci_list) / cv_fold, dtype='float'),
            'extended_ci': np.array(sum(extci_list) / cv_fold, dtype='float')
        }

            