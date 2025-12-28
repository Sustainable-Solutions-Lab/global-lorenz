"""
Lorenz curve functional forms and fitting routines.

This module provides three different functional forms for Lorenz curves:
1. 1-parameter form: General Quadratic (Ryu-Slottje)
2. 2-parameter form: Pareto-based
3. 3-parameter form: Generalized Beta of the Second Kind (GB2)
"""

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.special import beta as beta_func


def lorenz_1param(p, a):
    """
    1-parameter Lorenz curve (General Quadratic form).
    
    L(p) = p + a * p * (1 - p)
    
    Parameters:
    -----------
    p : array_like
        Cumulative proportion of population (0 to 1)
    a : float
        Shape parameter (typically between -1 and 1)
    
    Returns:
    --------
    L : array_like
        Cumulative proportion of income
    """
    p = np.asarray(p)
    return p + a * p * (1 - p)


def lorenz_2param(p, a, b):
    """
    2-parameter Lorenz curve (Pareto-based form).
    
    L(p) = 1 - (1 - p^a)^b
    
    Parameters:
    -----------
    p : array_like
        Cumulative proportion of population (0 to 1)
    a : float
        First shape parameter (a > 0)
    b : float
        Second shape parameter (b > 0)
    
    Returns:
    --------
    L : array_like
        Cumulative proportion of income
    """
    p = np.asarray(p)
    # Avoid numerical issues at boundaries
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return 1 - (1 - p**a)**b


def lorenz_3param(p, a, b, c):
    """
    3-parameter Lorenz curve (Generalized form).
    
    L(p) = p^a * (1 - (1-p)^b)^c
    
    Parameters:
    -----------
    p : array_like
        Cumulative proportion of population (0 to 1)
    a : float
        First shape parameter (a > 0)
    b : float
        Second shape parameter (b > 0)
    c : float
        Third shape parameter (c > 0)
    
    Returns:
    --------
    L : array_like
        Cumulative proportion of income
    """
    p = np.asarray(p)
    # Avoid numerical issues at boundaries
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return p**a * (1 - (1 - p)**b)**c


def gini_from_params(lorenz_func, params):
    """
    Calculate Gini coefficient from Lorenz curve parameters.
    
    Gini = 1 - 2 * integral(L(p) dp from 0 to 1)
    
    Parameters:
    -----------
    lorenz_func : callable
        Lorenz curve function
    params : tuple
        Parameters for the Lorenz curve
    
    Returns:
    --------
    gini : float
        Gini coefficient
    """
    from scipy.integrate import quad
    
    def integrand(p):
        return lorenz_func(p, *params)
    
    integral, _ = quad(integrand, 0, 1)
    gini = 1 - 2 * integral
    return gini


def fit_lorenz_curve(p_data, L_data, n_params=2, initial_guess=None):
    """
    Fit a Lorenz curve to data.
    
    Parameters:
    -----------
    p_data : array_like
        Cumulative proportion of population (sorted)
    L_data : array_like
        Cumulative proportion of income (corresponding to p_data)
    n_params : int
        Number of parameters (1, 2, or 3)
    initial_guess : tuple, optional
        Initial guess for parameters
    
    Returns:
    --------
    params : tuple
        Fitted parameters
    lorenz_func : callable
        The fitted Lorenz curve function
    gini : float
        Gini coefficient from fitted curve
    rmse : float
        Root mean squared error of fit
    """
    # Select the appropriate Lorenz curve form
    if n_params == 1:
        lorenz_func = lorenz_1param
        bounds = ([-1], [1])
        if initial_guess is None:
            initial_guess = [0.0]
    elif n_params == 2:
        lorenz_func = lorenz_2param
        bounds = ([0.1, 0.1], [5.0, 5.0])
        if initial_guess is None:
            initial_guess = [1.0, 1.0]
    elif n_params == 3:
        lorenz_func = lorenz_3param
        bounds = ([0.1, 0.1, 0.1], [5.0, 5.0, 5.0])
        if initial_guess is None:
            initial_guess = [1.0, 1.0, 1.0]
    else:
        raise ValueError("n_params must be 1, 2, or 3")
    
    # Remove boundary points if present (0, 0) and (1, 1)
    mask = (p_data > 0) & (p_data < 1)
    p_fit = p_data[mask]
    L_fit = L_data[mask]
    
    if len(p_fit) < n_params:
        raise ValueError(f"Not enough data points to fit {n_params} parameters")
    
    try:
        # Fit using curve_fit (least squares)
        params, _ = curve_fit(
            lorenz_func,
            p_fit,
            L_fit,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )
    except Exception as e:
        # If curve_fit fails, try optimization with constraints
        def objective(params):
            return np.sum((lorenz_func(p_fit, *params) - L_fit)**2)
        
        result = minimize(
            objective,
            initial_guess,
            bounds=[(bounds[0][i], bounds[1][i]) for i in range(n_params)],
            method='L-BFGS-B'
        )
        params = result.x
    
    # Calculate fit quality
    L_pred = lorenz_func(p_fit, *params)
    rmse = np.sqrt(np.mean((L_pred - L_fit)**2))
    
    # Calculate Gini coefficient
    gini = gini_from_params(lorenz_func, params)
    
    return tuple(params), lorenz_func, gini, rmse


def lorenz_to_quantile(lorenz_func, params, quantile):
    """
    Invert Lorenz curve to find income at a given quantile.
    
    Parameters:
    -----------
    lorenz_func : callable
        Lorenz curve function
    params : tuple
        Parameters for the Lorenz curve
    quantile : float
        Population quantile (0 to 1)
    
    Returns:
    --------
    income_fraction : float
        Fraction of mean income at this quantile
    """
    # For a given quantile q, the income at that point is:
    # dL/dp evaluated at p=q, divided by the mean income
    # This gives us income relative to mean
    
    epsilon = 1e-6
    q = np.clip(quantile, epsilon, 1 - epsilon)
    
    # Numerical derivative
    dp = 1e-6
    dL = lorenz_func(q + dp, *params) - lorenz_func(q - dp, *params)
    income_relative = dL / (2 * dp)
    
    return income_relative
