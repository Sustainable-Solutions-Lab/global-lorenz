"""
Country-level Lorenz curve fitting.

This module handles reading country-level income distribution data and
fitting Lorenz curves to each country.
"""

import numpy as np
import pandas as pd
from .lorenz_curves import fit_lorenz_curve


def read_country_data(filepath):
    """
    Read country-level income data from Excel or CSV file.

    Expected format:
    - country_name column
    - Income decile columns (decile1, decile2, ..., decile10)
    - reporting_gdp column (PPP GDP)
    - reporting_pop column (population)

    Parameters:
    -----------
    filepath : str
        Path to Excel or CSV file

    Returns:
    --------
    data : pandas DataFrame
        Country-level data with income distribution, GDP, and population
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    return df


def filter_most_recent_complete(data_df, income_cols):
    """
    Filter data to keep only the most recent year for each country with complete data.

    Parameters:
    -----------
    data_df : pandas DataFrame
        Country-level data with multiple years per country
    income_cols : list
        Column names containing income distribution data

    Returns:
    --------
    filtered_df : pandas DataFrame
        DataFrame with one row per country (most recent complete data)
    """
    required_cols = income_cols + ['reporting_pop', 'reporting_gdp']

    # Filter to rows with complete data
    complete_mask = data_df[required_cols].notna().all(axis=1)
    df_complete = data_df[complete_mask].copy()

    # Group by country and select most recent year
    df_recent = df_complete.sort_values('reporting_year', ascending=False).groupby('country_name').first().reset_index()

    return df_recent


def prepare_lorenz_data(income_shares):
    """
    Convert income shares to Lorenz curve data points.
    
    Parameters:
    -----------
    income_shares : array_like
        Income shares for each group (e.g., deciles)
        Should sum to approximately 1.0
    
    Returns:
    --------
    p : array
        Cumulative population proportions [0, 0.1, 0.2, ..., 1.0]
    L : array
        Cumulative income proportions [0, L1, L2, ..., 1.0]
    """
    income_shares = np.asarray(income_shares)
    
    # Normalize to ensure sum = 1
    income_shares = income_shares / np.sum(income_shares)
    
    # Number of groups
    n_groups = len(income_shares)
    
    # Population proportions (assuming equal groups)
    p = np.linspace(0, 1, n_groups + 1)
    
    # Cumulative income proportions
    L = np.concatenate([[0], np.cumsum(income_shares)])
    
    return p, L


def fit_country_lorenz_curves(data_df, income_cols, n_params=2, curve_type='quadratic'):
    """
    Fit Lorenz curves to all countries in the dataset.

    Parameters:
    -----------
    data_df : pandas DataFrame
        Country-level data
    income_cols : list
        Column names containing income distribution data (in order)
    n_params : int
        Number of parameters for Lorenz curve (1, 2, or 3)
    curve_type : str
        For n_params=3, select 'quadratic' (implicit quadratic), 'beta' (Kakwani beta), or 'sarabia' (ordered family)

    Returns:
    --------
    results : pandas DataFrame
        DataFrame with fitted parameters, Gini coefficients, and fit quality
    """
    results = []

    for idx, row in data_df.iterrows():
        country_name = row.get('country_name', row.get('Country', row.get('country', f'Country_{idx}')))

        # Extract income shares
        income_shares = row[income_cols].values

        # Skip if data is missing
        if np.any(pd.isna(income_shares)):
            print(f"Skipping {country_name}: missing income data")
            continue

        # Convert to Lorenz curve data
        p, L = prepare_lorenz_data(income_shares)

        try:
            # Fit Lorenz curve
            params, lorenz_func, gini, rmse = fit_lorenz_curve(p, L, n_params=n_params, curve_type=curve_type)

            result = {
                'country': country_name,
                'gini': gini,
                'rmse': rmse,
            }

            # Add parameters
            for i, param in enumerate(params):
                result[f'param_{i+1}'] = param

            # Add other data if available
            if 'reporting_gdp' in row.index:
                result['gdp'] = row['reporting_gdp']
            elif 'GDP' in row.index or 'gdp' in row.index:
                result['gdp'] = row.get('GDP', row.get('gdp'))

            if 'reporting_pop' in row.index:
                result['population'] = row['reporting_pop']
            elif 'Population' in row.index or 'population' in row.index:
                result['population'] = row.get('Population', row.get('population'))

            results.append(result)

        except Exception as e:
            print(f"Error fitting {country_name}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    return results_df


def evaluate_country_lorenz(row, n_params, income_thresholds, curve_type='quadratic'):
    """
    Evaluate a fitted country Lorenz curve at specific income thresholds.

    Parameters:
    -----------
    row : pandas Series
        Row from fitted results containing parameters
    n_params : int
        Number of parameters
    income_thresholds : array_like
        Income levels at which to evaluate (in units of mean income)
    curve_type : str
        For n_params=3: 'quadratic' or 'beta'

    Returns:
    --------
    populations_below : array
        Population fraction below each threshold
    """
    from .lorenz_curves import lorenz_pareto_1, lorenz_ortega_2, lorenz_gq_3, lorenz_beta_3, lorenz_sarabia_3

    # Extract parameters
    params = tuple(row[f'param_{i+1}'] for i in range(n_params))

    # Select appropriate function
    if n_params == 1:
        lorenz_func = lorenz_pareto_1
    elif n_params == 2:
        lorenz_func = lorenz_ortega_2
    elif n_params == 3:
        if curve_type == 'beta':
            lorenz_func = lorenz_beta_3
        elif curve_type == 'sarabia':
            lorenz_func = lorenz_sarabia_3
        else:
            lorenz_func = lorenz_gq_3
    
    # For each income threshold (as fraction of mean), find the population quantile
    # This requires inverting the derivative of the Lorenz curve
    # dL/dp = income at quantile p / mean income
    
    populations_below = []
    
    for threshold in income_thresholds:
        # Search for p where dL/dp = threshold
        # Use numerical search
        p_values = np.linspace(0.001, 0.999, 1000)
        dp = 0.001
        
        # Compute derivative numerically
        derivatives = np.zeros(len(p_values))
        for i, p in enumerate(p_values):
            dL = lorenz_func(p + dp/2, *params) - lorenz_func(p - dp/2, *params)
            derivatives[i] = dL / dp
        
        # Find where derivative crosses threshold
        if threshold < derivatives[0]:
            # Everyone below threshold
            pop_below = 0.0
        elif threshold > derivatives[-1]:
            # Everyone above threshold
            pop_below = 1.0
        else:
            # Interpolate
            pop_below = np.interp(threshold, derivatives, p_values)
        
        populations_below.append(pop_below)
    
    return np.array(populations_below)
