"""
Global aggregation and global Lorenz curve fitting.

This module aggregates country-level Lorenz curves to create a global
income distribution and fits a global Lorenz curve.
"""

import numpy as np
import pandas as pd
from .lorenz_curves import fit_lorenz_curve, lorenz_1param, lorenz_2param, lorenz_3param


def aggregate_global_distribution(country_results, n_params, 
                                   income_thresholds=None,
                                   gdp_col='gdp', pop_col='population'):
    """
    Aggregate country-level distributions to create global distribution.
    
    This function determines how many people worldwide have income below
    various threshold levels by using the fitted country-level Lorenz curves.
    
    Parameters:
    -----------
    country_results : pandas DataFrame
        Results from fit_country_lorenz_curves with parameters, GDP, and population
    n_params : int
        Number of parameters used in country fits
    income_thresholds : array_like, optional
        Income levels to evaluate (in PPP dollars/year)
        If None, creates a range from min to max
    gdp_col : str
        Column name for GDP data
    pop_col : str
        Column name for population data
    
    Returns:
    --------
    global_data : pandas DataFrame
        DataFrame with income thresholds and global cumulative distribution
    """
    from .country_fitting import evaluate_country_lorenz
    
    # Calculate mean income per capita for each country (GDP / population)
    country_results = country_results.copy()
    country_results['mean_income'] = country_results[gdp_col] / country_results[pop_col]
    
    # If no thresholds provided, create a range
    if income_thresholds is None:
        min_income = country_results['mean_income'].min() * 0.1
        max_income = country_results['mean_income'].max() * 2.0
        income_thresholds = np.logspace(
            np.log10(min_income),
            np.log10(max_income),
            100
        )
    
    income_thresholds = np.asarray(income_thresholds)
    
    # For each income threshold, calculate how many people globally are below it
    global_pop_below = np.zeros(len(income_thresholds))
    total_population = country_results[pop_col].sum()
    
    for idx, row in country_results.iterrows():
        # Convert absolute income thresholds to fractions of country mean
        country_mean = row['mean_income']
        threshold_fractions = income_thresholds / country_mean
        
        # Get population fractions below each threshold for this country
        try:
            pop_fractions = evaluate_country_lorenz(row, n_params, threshold_fractions)
            
            # Convert to absolute numbers and add to global total
            country_pop = row[pop_col]
            global_pop_below += pop_fractions * country_pop
            
        except Exception as e:
            print(f"Error evaluating country {row['country']}: {e}")
            continue
    
    # Convert to cumulative fractions
    global_pop_fraction = global_pop_below / total_population
    
    # Create DataFrame
    global_data = pd.DataFrame({
        'income_threshold': income_thresholds,
        'population_below': global_pop_below,
        'population_fraction': global_pop_fraction
    })
    
    # Sort by income threshold
    global_data = global_data.sort_values('income_threshold').reset_index(drop=True)
    
    return global_data


def global_distribution_to_lorenz(global_data):
    """
    Convert global income distribution to Lorenz curve format.
    
    Parameters:
    -----------
    global_data : pandas DataFrame
        Output from aggregate_global_distribution
    
    Returns:
    --------
    p : array
        Cumulative population fractions
    L : array
        Cumulative income fractions
    """
    # Sort by income threshold
    data = global_data.sort_values('income_threshold').copy()
    
    # Population fractions (p)
    p = data['population_fraction'].values
    
    # To get L (cumulative income fraction), we need to integrate
    # income * population over the distribution
    
    # Use trapezoidal rule to compute cumulative income
    income = data['income_threshold'].values
    pop = data['population_below'].values
    
    # Total income is approximately integral of income * d(population)
    # We'll compute this numerically
    
    # Income at each point times the population increment
    dpop = np.diff(pop)
    income_mid = (income[:-1] + income[1:]) / 2
    
    # Cumulative income
    income_increments = income_mid * dpop
    cumulative_income = np.concatenate([[0], np.cumsum(income_increments)])
    
    # Normalize to get L
    total_income = cumulative_income[-1]
    if total_income > 0:
        L = cumulative_income / total_income
    else:
        L = p.copy()
    
    # Ensure L is properly bounded and monotonic
    L = np.clip(L, 0, 1)
    L = np.maximum.accumulate(L)
    
    return p, L


def fit_global_lorenz(country_results, n_params_country, n_params_global=2,
                     income_thresholds=None):
    """
    Fit a global Lorenz curve from country-level data.
    
    Parameters:
    -----------
    country_results : pandas DataFrame
        Results from fit_country_lorenz_curves
    n_params_country : int
        Number of parameters used in country-level fits
    n_params_global : int
        Number of parameters for global Lorenz curve (1, 2, or 3)
    income_thresholds : array_like, optional
        Income levels to evaluate for aggregation
    
    Returns:
    --------
    global_params : tuple
        Fitted parameters for global Lorenz curve
    global_lorenz_func : callable
        The fitted global Lorenz curve function
    global_gini : float
        Global Gini coefficient
    global_data : pandas DataFrame
        Global income distribution data
    """
    # Aggregate to global distribution
    global_data = aggregate_global_distribution(
        country_results,
        n_params_country,
        income_thresholds=income_thresholds
    )
    
    # Convert to Lorenz curve format
    p, L = global_distribution_to_lorenz(global_data)
    
    # Fit global Lorenz curve
    global_params, global_lorenz_func, global_gini, rmse = fit_lorenz_curve(
        p, L, n_params=n_params_global
    )
    
    print(f"Global Lorenz curve fitted with {n_params_global} parameters")
    print(f"Global Gini coefficient: {global_gini:.4f}")
    print(f"RMSE: {rmse:.6f}")
    
    return global_params, global_lorenz_func, global_gini, global_data


def compute_global_poverty_metrics(country_results, n_params, poverty_lines):
    """
    Compute global poverty metrics at various poverty lines.
    
    Parameters:
    -----------
    country_results : pandas DataFrame
        Results from fit_country_lorenz_curves
    n_params : int
        Number of parameters used in country fits
    poverty_lines : array_like
        Poverty line thresholds (in PPP dollars/year)
    
    Returns:
    --------
    metrics : pandas DataFrame
        Poverty headcount and other metrics for each poverty line
    """
    from .country_fitting import evaluate_country_lorenz
    
    poverty_lines = np.asarray(poverty_lines)
    total_population = country_results['population'].sum()
    
    results = []
    
    for poverty_line in poverty_lines:
        people_below = 0
        
        for idx, row in country_results.iterrows():
            country_mean = row['gdp'] / row['population']
            threshold_fraction = poverty_line / country_mean
            
            try:
                pop_fractions = evaluate_country_lorenz(row, n_params, [threshold_fraction])
                people_below += pop_fractions[0] * row['population']
            except:
                continue
        
        results.append({
            'poverty_line': poverty_line,
            'people_below': people_below,
            'poverty_rate': people_below / total_population
        })
    
    return pd.DataFrame(results)
