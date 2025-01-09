import numpy as np


def cross_freq_list(x, y, x_label=[0, 1], y_label=[0, 1]):
    """
    Computing the cross frequency table.

    Args:
        x (list or np.ndarray): A 1D array of binary elements.
        y (list or np.ndarray): A 1D array of binary elements.
        x_label (list or np.ndarray, optional): Labels of x. Default is [0, 1].
        y_label (list or np.ndarray, optional): Labels of y. Default is [0, 1].

    Returns:
        np.ndarray: A 2x2 contingency table where each element represents the 
        frequency of a specific combination of x and y values.

    """
    # Validate x_label and y_label
    if not isinstance(x_label, (list, np.ndarray)) or not isinstance(
            y_label, (list, np.ndarray)):
        raise ValueError("x_label and y_label must be lists or numpy arrays.")
    if len(x_label) * len(y_label) != 4:
        raise ValueError(
            "x_label and y_label must have lengths that allow reshaping to\
            (2, 2)."
            )

    return np.reshape([sum((x==k)&(y==l)) for k in x_label for l in y_label], (2, 2))


def convert_to_np(*args):
    """
    Convert the input data to numpy arrays. If the input array has shape (*, 1), 
    it will be flattened to (*,).

    Args:
        *args (list, pd.DataFrame, or np.ndarray): One or more input data 
        structures to be converted.

    Returns:
        list: A list of numpy arrays converted from the inputs. Each array has 
        appropriate dimensions.

    """
    result = []
    for arg in args:
        arr = np.array(arg)
        
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.flatten()
        
        result.append(arr)
    
    return result


def filter_data(**kwargs):
    """
    Filters data based on treatment groups and response levels. Optionally 
    filters feature matrix.

    Args:
        kwargs (dict): Keyword arguments specifying input data and parameters. 
        Expected keys:
            - X (np.ndarray or list, optional): Feature matrix.
            - treatment (np.ndarray or list): Treatment assignment vector.
            - y (np.ndarray or list): Response vector.
            - treated_arms (list or np.ndarray): Labels for treated group.
            - control_arms (list or np.ndarray): Labels for control group.
            - y_level (list or np.ndarray): Allowed response levels.

    Returns:
        tuple: Depending on input:
            - If X is provided: (X_filt, treatment_filt, y_filt).
            - If X is not provided: (treatment_filt, y_filt).

    Raises:
        ValueError: If required arguments are missing or invalid.

    """
    required_keys = ['treatment', 'y', 'treated_arms', 'control_arms', 'y_level']
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required argument: {key}")

    X = kwargs.get('X', None)
    treatment = np.array(kwargs['treatment'])
    y = np.array(kwargs['y'])
    treated_arms = kwargs['treated_arms']
    control_arms= kwargs['control_arms']
    y_level = kwargs['y_level']

    mask = (np.isin(treatment, np.hstack([treated_arms, control_arms]))) & (np.isin(y, y_level))
    mask = mask.flatten()
    treatment_filt = np.isin(treatment[mask], treated_arms).astype(int)
    treatment_filt = treatment_filt.flatten()
    y_filt = y[mask]
    y_filt = y_filt.flatten()

    if X is not None:
        X = np.array(X)
        X_filt = X[mask]
        if X_filt.ndim == 1:
            X_filt = X_filt.reshape(-1, 1)
        return X_filt, treatment_filt, y_filt

    return treatment_filt, y_filt

def safe_div(x, y):
    """
    Safely performs division and handles division by zero.

    Args:
        x (float): Numerator.
        y (float): Denominator.

    Returns:
        float: Result of division if y != 0, otherwise 0.

    """
    if y==0:
        return 0
    else:
        return x/y
    
def Minrow_idx(*args):   
    """
    Finds the indices of the minimum values along the rows of combined arrays.

    Args:
        *args (np.ndarray): Input arrays to be combined.

    Returns:
        np.ndarray: Indices of minimum values along axis 1.

    """
    combined = np.hstack(args)
    indices = np.argmin(combined, axis=1)
    return indices

def Mincol_idx(*args):   
    """
    Finds the indices of the minimum values along the columns of combined arrays.

    Args:
        *args (np.ndarray): Input arrays to be combined.

    Returns:
        np.ndarray: Indices of minimum values along axis 0.

    """
    combined = np.stack(args)
    indices = np.argmin(combined, axis=0)
    return indices


def Minrow_value(*args):   
    """
    Finds the indices of the minimum values along the rows of combined arrays.

    Args:
        *args (np.ndarray): Input arrays to be combined.

    Returns:
        np.ndarray: Indices of minimum values along axis 1.

    """
    combined = np.hstack(args)
    values = np.min(combined, axis=1)
    return values