"""
Lorenz curve functional forms and fitting routines.

This module provides three different functional forms for Lorenz curves:
1. 1-parameter form: Pareto Lorenz
2. 2-parameter form: Jantzen-Volpert form
3. 3-parameter form: Generalized-Quadratic (GQ) form
"""

import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.integrate import quad

# Small epsilon for numerical stability in parameter bounds
EPSILON = 1e-12


def lorenz_pareto_1(p, a):
    """
    1-parameter Lorenz curve (Pareto Lorenz).

    L(p) = 1 - (1 - p)^(1 - 1/a)

    Reference:
    https://www.mdpi.com/2225-1146/13/3/30

    Parameters:
    -----------
    p : array_like
        Cumulative proportion of population (0 to 1)
    a : float
        Shape parameter for the Pareto distribution (a > 1)

    Note: The Gini index for this form is G = 1/(2a - 1)

    Returns:
    --------
    L : array_like
        Cumulative proportion of income
    """
    p = np.asarray(p)
    return 1 - (1 - p)**(1 - 1/a)


def lorenz_ortega_2(p, a, b):
    """
    2-parameter Lorenz curve (Ortega/Jantzen-Volpert functional form).

    L(p) = p^a * (1 - (1-p)^b)

    Constraints: a >= 0, 0 < b <= 1

    References:
    Ortega, P., M.A. Fernández, M. Ladoux, A. García (1991). A new functional form for
    estimating Lorenz curves. Review of Income and Wealth, 37, 447-452.
    https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1475-4991.1991.tb00383.x

    Parameters:
    -----------
    p : array_like
        Cumulative proportion of population (0 to 1)
    a : float
        First shape parameter (a >= 0)
    b : float
        Second shape parameter (0 < b <= 1)

    Returns:
    --------
    L : array_like
        Cumulative proportion of income
    """
    p = np.asarray(p)
    return p**a * (1 - (1 - p)**b)


def lorenz_gq_3(p, a, b, c):
    """
    3-parameter Generalized Quadratic Lorenz curve solving the implicit equation:
    L(1-L) = a(p² - L) + bL(p-1) + c(p-L)

    This reduces to a quadratic equation in L:
    L² - L(1 + a + b + c - bp) + ap² + cp = 0

    References:
    Villasenor, J., and B. Arnold (1989). Elliptical Lorenz curves. Journal of Econometrics, 40, 327–338.
    https://rpubs.com/tsamuel/709170

    Parameters:
    -----------
    p : array_like
        Cumulative proportion of population (0 to 1)
    a : float
        First parameter
    b : float
        Second parameter
    c : float
        Third parameter

    Returns:
    --------
    L : array_like
        Cumulative proportion of income (the solution between 0 and 1)
    """
    p = np.asarray(p)

    # Quadratic equation coefficients: A*L² + B*L + C = 0
    A = 1.0
    B = -(1 + a + b + c - b * p)
    C = a * p**2 + c * p

    # Discriminant
    discriminant = B**2 - 4 * A * C

    # Ensure non-negative discriminant
    discriminant = np.maximum(discriminant, 0)

    # Two solutions from quadratic formula
    sqrt_disc = np.sqrt(discriminant)
    L1 = (-B + sqrt_disc) / (2 * A)
    L2 = (-B - sqrt_disc) / (2 * A)

    # Select the solution that lies between 0 and 1
    L = np.where((L1 >= 0) & (L1 <= 1), L1, L2)

    # Clip to ensure we stay in valid range
    L = np.clip(L, 0, 1)

    return L


def lorenz_sarabia_3(p, a, b, c):
    """
    Sarabia 3-parameter Lorenz curve (Ordered Family form).

    L(p) = p^a · (1 - (1-p)^b)^c

    Constraints: a > 0, b > 0, c > 1

    Reference:
    Sarabia, J., E. Castillo and D. Slottje (1999). An Ordered Family of Lorenz Curves,
    Journal of Econometrics, 91, 43-60.

    Parameters:
    -----------
    p : array_like
        Cumulative proportion of population (0 to 1)
    a : float
        First parameter (a > 0)
    b : float
        Second parameter (b > 0)
    c : float
        Third parameter (c > 1)

    Returns:
    --------
    L : array_like
        Cumulative proportion of income
    """
    p = np.asarray(p)
    # Avoid numerical issues at boundaries
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return p**a * (1 - (1 - p)**b)**c


def lorenz_beta_3(p, a, b, c):
    """
    Beta Lorenz curve (3-parameter form).

    L(p) = p - a * p^b * (1-p)^c

    Reference:
    Kakwani, N. (1980). On a Class of Poverty Measures, Econometrica, 48, 437–446.

    Parameters:
    -----------
    p : array_like
        Cumulative proportion of population (0 to 1)
    a : float
        First parameter (0 < a < 1)
    b : float
        Second parameter (0 < b < 1)
    c : float
        Third parameter (0 < c < 1)

    Returns:
    --------
    L : array_like
        Cumulative proportion of income
    """
    p = np.asarray(p)
    return p - a * p**b * (1 - p)**c


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
    def integrand(p):
        return lorenz_func(p, *params)
    
    integral, _ = quad(integrand, 0, 1)
    gini = 1 - 2 * integral
    return gini


def fit_lorenz_curve_decile(income_shares, lorenz_type):
    """
    Fit a Lorenz curve to decile income share data using squared fractional error objective.

    Minimizes sum over N=1 to num_deciles of: ((L(N/num_deciles) - L((N-1)/num_deciles)) / share_N - 1)^2

    This objective function gives equal weight to all deciles regardless of their absolute size.

    Parameters:
    -----------
    income_shares : array_like
        Income shares for each decile/bin (should sum to approximately 1)
    lorenz_type : str
        Lorenz curve type: 'pareto_1', 'ortega_2', 'gq_3', 'beta_3', or 'sarabia_3'

    Returns:
    --------
    params : tuple
        Fitted parameters
    lorenz_func : callable
        The fitted Lorenz curve function
    gini : float
        Gini coefficient from fitted curve
    rmse : float
        Root mean squared fractional error
    mafe : float
        Mean absolute fractional error
    max_abs_error : float
        Maximum absolute fractional error across all bins
    """
    income_shares = np.asarray(income_shares)
    income_shares = income_shares / np.sum(income_shares)
    n_bins = len(income_shares)

    # Extract n_params from lorenz_type
    n_params = int(lorenz_type.split('_')[1])

    # Select the appropriate Lorenz curve form based on lorenz_type
    if lorenz_type == 'pareto_1':
        lorenz_func = lorenz_pareto_1
        bounds = ([1.0 + EPSILON], [20.0])
        initial_guess = [2.0]
    elif lorenz_type == 'ortega_2':
        lorenz_func = lorenz_ortega_2
        bounds = ([0.0, EPSILON], [5.0, 1.0])
        initial_guess = [1.0, 0.5]
    elif lorenz_type == 'gq_3':
        lorenz_func = lorenz_gq_3
        bounds = ([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0])
        initial_guess = [0.5, 0.5, 0.5]
    elif lorenz_type == 'beta_3':
        lorenz_func = lorenz_beta_3
        bounds = ([EPSILON, EPSILON, EPSILON], [1-EPSILON, 1-EPSILON, 1-EPSILON])
        initial_guess = [0.5, 0.5, 0.5]
    elif lorenz_type == 'sarabia_3':
        lorenz_func = lorenz_sarabia_3
        bounds = ([EPSILON, EPSILON, 1+EPSILON], [5.0, 5.0, 5.0])
        initial_guess = [0.2, 0.66667, 1.3]
    else:
        raise ValueError(f"Unknown lorenz_type: {lorenz_type}. Must be one of: 'pareto_1', 'ortega_2', 'gq_3', 'beta_3', 'sarabia_3'")

    # Define fractional error objective function
    def objective(params):
        total_squared_error = 0.0
        for i in range(n_bins):
            p_lower = i / n_bins
            p_upper = (i + 1) / n_bins

            L_lower = lorenz_func(p_lower, *params)
            L_upper = lorenz_func(p_upper, *params)

            predicted_share = L_upper - L_lower
            actual_share = income_shares[i]

            fractional_error = (predicted_share / actual_share) - 1.0
            total_squared_error += fractional_error ** 2

        return total_squared_error

    # Optimize using L-BFGS-B with bounds
    result = minimize(
        objective,
        initial_guess,
        bounds=[(bounds[0][i], bounds[1][i]) for i in range(n_params)],
        method='L-BFGS-B'
    )
    params = tuple(result.x)

    # Calculate goodness-of-fit statistics
    fractional_errors = []
    for i in range(n_bins):
        p_lower = i / n_bins
        p_upper = (i + 1) / n_bins
        L_lower = lorenz_func(p_lower, *params)
        L_upper = lorenz_func(p_upper, *params)
        predicted_share = L_upper - L_lower
        actual_share = income_shares[i]
        fractional_error = (predicted_share / actual_share) - 1.0
        fractional_errors.append(fractional_error)

    fractional_errors = np.array(fractional_errors)

    rmse = np.sqrt(np.mean(fractional_errors ** 2))
    mafe = np.mean(np.abs(fractional_errors))
    max_abs_error = np.max(np.abs(fractional_errors))

    # Calculate Gini coefficient
    gini = gini_from_params(lorenz_func, params)

    return params, lorenz_func, gini, rmse, mafe, max_abs_error


def fit_lorenz_curve(p_data, L_data, lorenz_type):
    """
    Fit a Lorenz curve to data.

    Parameters:
    -----------
    p_data : array_like
        Cumulative proportion of population (sorted)
    L_data : array_like
        Cumulative proportion of income (corresponding to p_data)
    lorenz_type : str
        Lorenz curve type: 'pareto_1', 'ortega_2', 'gq_3', 'beta_3', or 'sarabia_3'

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
    # Extract n_params from lorenz_type
    n_params = int(lorenz_type.split('_')[1])

    # Select the appropriate Lorenz curve form based on lorenz_type
    if lorenz_type == 'pareto_1':
        lorenz_func = lorenz_pareto_1
        bounds = ([1.0 + EPSILON], [5.0])  # maximum for real data is about 2.5
        initial_guess = [2.0]
    elif lorenz_type == 'ortega_2':
        lorenz_func = lorenz_ortega_2
        bounds = ([0.0, EPSILON], [1.0, 1.0])
        initial_guess = [0.33333, 0.66667]
    elif lorenz_type == 'gq_3':
        lorenz_func = lorenz_gq_3
        bounds = ([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0])
        initial_guess = [0.5, 0.5, 0.5]
    elif lorenz_type == 'beta_3':
        lorenz_func = lorenz_beta_3
        bounds = ([EPSILON, EPSILON, EPSILON], [1-EPSILON, 1-EPSILON, 1-EPSILON])
        initial_guess = [0.5, 0.5, 0.5]
    elif lorenz_type == 'sarabia_3':
        lorenz_func = lorenz_sarabia_3
        bounds = ([EPSILON, EPSILON, 1+EPSILON], [5.0, 5.0, 5.0])
        initial_guess = [0.2, 0.66667, 1.33333]
    else:
        raise ValueError(f"Unknown lorenz_type: {lorenz_type}. Must be one of: 'pareto_1', 'ortega_2', 'gq_3', 'beta_3', 'sarabia_3'")
    
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
