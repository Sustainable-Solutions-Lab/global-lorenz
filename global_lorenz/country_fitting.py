"""
Country-level Lorenz curve fitting.

This module handles reading country-level income distribution data and
fitting Lorenz curves to each country.
"""

import numpy as np
import pandas as pd
from .lorenz_curves import fit_lorenz_curve_decile, lorenz_pareto_1, lorenz_ortega_2, lorenz_gq_3, lorenz_beta_3, lorenz_sarabia_3


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


def fit_country_lorenz_curves(data_df, income_cols, lorenz_type):
    """
    Fit Lorenz curves to all countries in the dataset.

    Parameters:
    -----------
    data_df : pandas DataFrame
        Country-level data
    income_cols : list
        Column names containing income distribution data (in order)
    lorenz_type : str
        Lorenz curve type: 'pareto_1', 'ortega_2', 'gq_3', 'beta_3', or 'sarabia_3'

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

        try:
            # Fit Lorenz curve using hybrid error objective
            (params, lorenz_func, gini, rmse, mafe, max_abs_error,
             fractional_rmse, fractional_mafe, fractional_max_abs_error,
             absolute_rmse, absolute_mafe, absolute_max_abs_error) = fit_lorenz_curve_decile(income_shares, lorenz_type)

            result = {
                'country': country_name,
                'gini': gini,
                'rmse': rmse,
                'mafe': mafe,
                'max_abs_error': max_abs_error,
                'fractional_rmse': fractional_rmse,
                'fractional_mafe': fractional_mafe,
                'fractional_max_abs_error': fractional_max_abs_error,
                'absolute_rmse': absolute_rmse,
                'absolute_mafe': absolute_mafe,
                'absolute_max_abs_error': absolute_max_abs_error,
            }

            # Add parameters
            for i, param in enumerate(params):
                result[f'param_{i+1}'] = param

            # Add year
            if 'reporting_year' in row.index:
                result['year'] = row['reporting_year']

            # Add other data if available
            if 'reporting_gdp' in row.index:
                result['gdp'] = row['reporting_gdp']
            elif 'GDP' in row.index or 'gdp' in row.index:
                result['gdp'] = row.get('GDP', row.get('gdp'))

            if 'reporting_pop' in row.index:
                result['population'] = row['reporting_pop']
            elif 'Population' in row.index or 'population' in row.index:
                result['population'] = row.get('Population', row.get('population'))

            # Add actual income shares
            n_bins = len(income_shares)
            for i in range(n_bins):
                result[f'decile{i+1}_actual'] = income_shares[i]

            # Add fitted income shares
            for i in range(n_bins):
                p_lower = i / n_bins
                p_upper = (i + 1) / n_bins
                L_lower = lorenz_func(p_lower, *params)
                L_upper = lorenz_func(p_upper, *params)
                predicted_share = L_upper - L_lower
                result[f'decile{i+1}_fitted'] = predicted_share

            results.append(result)

        except Exception as e:
            print(f"Error fitting {country_name}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    return results_df


def evaluate_country_lorenz(row, income_thresholds, lorenz_type):
    """
    Evaluate a fitted country Lorenz curve at specific income thresholds.

    Parameters:
    -----------
    row : pandas Series
        Row from fitted results containing parameters
    income_thresholds : array_like
        Income levels at which to evaluate (in units of mean income)
    lorenz_type : str
        Lorenz curve type: 'pareto_1', 'ortega_2', 'gq_3', 'beta_3', or 'sarabia_3'

    Returns:
    --------
    populations_below : array
        Population fraction below each threshold
    """
    # Extract n_params from lorenz_type
    n_params = int(lorenz_type.split('_')[1])

    # Extract parameters
    params = tuple(row[f'param_{i+1}'] for i in range(n_params))

    # Select appropriate function based on lorenz_type
    if lorenz_type == 'pareto_1':
        lorenz_func = lorenz_pareto_1
    elif lorenz_type == 'ortega_2':
        lorenz_func = lorenz_ortega_2
    elif lorenz_type == 'gq_3':
        lorenz_func = lorenz_gq_3
    elif lorenz_type == 'beta_3':
        lorenz_func = lorenz_beta_3
    elif lorenz_type == 'sarabia_3':
        lorenz_func = lorenz_sarabia_3
    else:
        raise ValueError(f"Unknown lorenz_type: {lorenz_type}")
    
    # For each income threshold (as fraction of mean), find the population quantile
    # This requires inverting the derivative of the Lorenz curve
    # dL/dp = income at quantile p / mean income

    # Compute derivative grid ONCE for this country (not per threshold - major speedup!)
    p_values = np.linspace(0.001, 0.999, 1000)
    dp = 0.001

    # Vectorized derivative computation
    p_plus = p_values + dp/2
    p_minus = p_values - dp/2
    L_plus = lorenz_func(p_plus, *params)
    L_minus = lorenz_func(p_minus, *params)
    derivatives = (L_plus - L_minus) / dp

    # Vectorized threshold lookup using numpy's interp (handles arrays)
    income_thresholds = np.asarray(income_thresholds)
    populations_below = np.interp(income_thresholds, derivatives, p_values,
                                   left=0.0, right=1.0)

    return populations_below
