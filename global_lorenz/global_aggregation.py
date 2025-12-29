"""
Global aggregation and global Lorenz curve fitting.

This module aggregates country-level Lorenz curves to create a global
income distribution and fits a global Lorenz curve.
"""

import numpy as np
import pandas as pd
from .lorenz_curves import lorenz_pareto_1, lorenz_ortega_2, lorenz_gq_3, lorenz_beta_3, lorenz_sarabia_3
from .country_fitting import evaluate_country_lorenz


def aggregate_global_distribution(country_results,
                                   lorenz_type,
                                   income_thresholds,
                                   gdp_col, pop_col,
                                   n_agg_bins=1000):
    """
    Aggregate country-level distributions to create global distribution.

    This function determines how many people worldwide have income below
    various threshold levels by using the fitted country-level Lorenz curves.

    Parameters:
    -----------
    country_results : pandas DataFrame
        Results from fit_country_lorenz_curves with parameters, GDP, and population
    lorenz_type : str
        Lorenz curve type: 'pareto_1', 'ortega_2', 'gq_3', 'beta_3', or 'sarabia_3'
    income_thresholds : array_like
        Income levels to evaluate (in PPP dollars/year)
        If None, creates a range from min to max
    gdp_col : str
        Column name for GDP data
    pop_col : str
        Column name for population data
    n_agg_bins : int, optional
        Number of bins for high-resolution aggregation [default: 1000]

    Returns:
    --------
    global_data : pandas DataFrame
        DataFrame with income thresholds and global cumulative distribution
    """
    # Note: gdp_col is already GDP per capita (PPP-adjusted), not total GDP
    country_results = country_results.copy()
    country_results['mean_income'] = country_results[gdp_col]
    
    # If no thresholds provided, create a range
    if income_thresholds is None:
        min_income = country_results['mean_income'].min() * 0.1
        max_income = country_results['mean_income'].max() * 2.0
        income_thresholds = np.logspace(
            np.log10(min_income),
            np.log10(max_income),
            n_agg_bins  # High resolution for accurate aggregation
        )
    
    income_thresholds = np.asarray(income_thresholds)
    
    # For each income threshold, calculate how many people globally are below it
    # AND how much income they have
    global_pop_below = np.zeros(len(income_thresholds))
    global_income_below = np.zeros(len(income_thresholds))
    total_population = country_results[pop_col].sum()

    # Select appropriate Lorenz function
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

    n_params = int(lorenz_type.split('_')[1])

    for idx, row in country_results.iterrows():
        # Convert absolute income thresholds to fractions of country mean
        country_mean = row['mean_income']
        country_pop = row[pop_col]
        country_total_income = country_mean * country_pop
        threshold_fractions = income_thresholds / country_mean

        # Get population fractions below each threshold for this country
        try:
            pop_fractions = evaluate_country_lorenz(row, threshold_fractions, lorenz_type)

            # Get income fractions using the country's Lorenz curve
            params = tuple(row[f'param_{i+1}'] for i in range(n_params))
            income_fractions = lorenz_func(pop_fractions, *params)

            # Add to global totals
            global_pop_below += pop_fractions * country_pop
            global_income_below += income_fractions * country_total_income

        except Exception as e:
            print(f"Error evaluating country {row['country']}: {e}")
            continue

    # Convert to cumulative fractions
    total_income = global_income_below[-1] if len(global_income_below) > 0 else 1.0
    global_pop_fraction = global_pop_below / total_population
    global_income_fraction = global_income_below / total_income

    # Create DataFrame
    global_data = pd.DataFrame({
        'income_threshold': income_thresholds,
        'population_below': global_pop_below,
        'population_fraction': global_pop_fraction,
        'income_below': global_income_below,
        'income_fraction': global_income_fraction
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
        Output from aggregate_global_distribution with columns:
        - population_fraction: cumulative population fraction (p)
        - income_fraction: cumulative income fraction (L)

    Returns:
    --------
    p : array
        Cumulative population fractions
    L : array
        Cumulative income fractions
    """
    # Sort by income threshold (should already be sorted, but be safe)
    data = global_data.sort_values('income_threshold').copy()

    # Extract cumulative fractions (already computed correctly)
    p = data['population_fraction'].values
    L = data['income_fraction'].values

    # Ensure L is properly bounded and monotonic
    L = np.clip(L, 0, 1)
    L = np.maximum.accumulate(L)

    return p, L


def resample_to_equal_population_bins(p, L, n_bins):
    """
    Resample Lorenz curve data to equal-population bins.

    This converts high-resolution (p, L) data with unequal bin sizes to
    lower-resolution data with equal population bins, which works better
    for fractional error fitting.

    Parameters:
    -----------
    p : array
        Cumulative population fractions (sorted, 0 to 1)
    L : array
        Cumulative income fractions (corresponding to p)
    n_bins : int
        Number of equal-population bins to create

    Returns:
    --------
    income_shares : array
        Income share in each bin (length n_bins)
    population_shares : array
        Population share in each bin (length n_bins, all equal to 1/n_bins)
    """
    # Create target population fractions at bin boundaries
    p_target = np.linspace(0, 1, n_bins + 1)

    # Interpolate L at target population fractions
    L_target = np.interp(p_target, p, L)

    # Compute income shares and population shares
    income_shares = np.diff(L_target)
    population_shares = np.diff(p_target)

    return income_shares, population_shares


def fit_global_lorenz(country_results, lorenz_type, income_thresholds, error_type='hybrid', fit_on_cumulative=False,
                      n_agg_bins=1000, n_fit_bins=100):
    """
    Fit a global Lorenz curve from country-level data.

    Parameters:
    -----------
    country_results : pandas DataFrame
        Results from fit_country_lorenz_curves
    lorenz_type : str
        Lorenz curve type: 'pareto_1', 'ortega_2', 'gq_3', 'beta_3', or 'sarabia_3'
    income_thresholds : array_like
        Income levels to evaluate for aggregation
    error_type : str, optional
        Error metric for fitting [default: 'hybrid']
        Options: 'hybrid', 'absolute', 'fractional'
    fit_on_cumulative : bool, optional
        If True, fit on cumulative L(p) values [default: False]
    n_agg_bins : int, optional
        Number of bins for high-resolution aggregation [default: 1000]
    n_fit_bins : int, optional
        Number of equal-population bins for fitting [default: 100]

    Returns:
    --------
    global_params : tuple
        Fitted parameters for global Lorenz curve
    global_lorenz_func : callable
        The fitted global Lorenz curve function
    global_gini : float
        Global Gini coefficient
    rmse : float
        Root mean squared hybrid error
    mafe : float
        Mean absolute hybrid error
    max_abs_error : float
        Maximum absolute hybrid error
    fractional_rmse : float
        Root mean squared fractional error
    fractional_mafe : float
        Mean absolute fractional error
    fractional_max_abs_error : float
        Maximum absolute fractional error
    absolute_rmse : float
        Root mean squared absolute error
    absolute_mafe : float
        Mean absolute absolute error
    absolute_max_abs_error : float
        Maximum absolute absolute error
    global_data : pandas DataFrame
        Global income distribution data with columns:
        - income_threshold: Income level (PPP dollars/year)
        - population_below: Number of people below threshold
        - population_fraction: Cumulative population fraction (actual, aggregated)
        - income_below: Total income below threshold
        - income_fraction: Cumulative income fraction (actual, aggregated)
        - income_fraction_fitted: Cumulative income fraction (fitted Lorenz curve)
    fitted_bins_data : pandas DataFrame
        Data for the 100 equal-population bins used for fitting with columns:
        - bin_number: Bin index (0-99)
        - population_share: Population share in this bin
        - population_cumulative: Cumulative population fraction at bin end
        - income_share_actual: Actual income share in this bin
        - income_cumulative_actual: Actual cumulative income fraction at bin end
        - income_share_fitted: Fitted income share in this bin
        - income_cumulative_fitted: Fitted cumulative income fraction at bin end
    """
    # Aggregate to global distribution
    global_data = aggregate_global_distribution(
        country_results,
        lorenz_type,
        income_thresholds,
        'gdp',
        'population',
        n_agg_bins=n_agg_bins,
    )

    # Convert to Lorenz curve format (high-resolution)
    p, L = global_distribution_to_lorenz(global_data)

    # Resample to equal-population bins
    # This provides good resolution while avoiding tiny bins at extremes
    income_shares, population_shares = resample_to_equal_population_bins(p, L, n_bins=n_fit_bins)

    # Fit global Lorenz curve using HYBRID error
    # This minimizes the product of absolute and relative errors: (pred - actual)² / actual
    # Small bins get upweighted (divide by small actual), large bins downweighted
    #
    # Error type options:
    # - 'hybrid':       minimize (pred-actual)²/actual [RECOMMENDED - balanced weighting]
    # - 'intermediate': weight by pop * income, absolute error
    # - 'absolute':     weight by pop, absolute error
    # - 'fractional':   weight by pop, fractional error
    from .lorenz_curves import fit_lorenz_curve_decile
    (global_params, global_lorenz_func, global_gini, rmse, mafe, max_abs_error,
     fractional_rmse, fractional_mafe, fractional_max_abs_error,
     absolute_rmse, absolute_mafe, absolute_max_abs_error) = fit_lorenz_curve_decile(
        income_shares, lorenz_type, population_shares, error_type=error_type, fit_on_cumulative=fit_on_cumulative
    )
    # # Other error type options (commented out):
    # global_params, global_lorenz_func, global_gini, rmse, mafe, max_abs_error = fit_lorenz_curve_decile(
    #     income_shares, lorenz_type, population_shares, error_type='intermediate'
    # )
    # global_params, global_lorenz_func, global_gini, rmse, mafe, max_abs_error = fit_lorenz_curve_decile(
    #     income_shares, lorenz_type, population_shares, error_type='absolute'
    # )
    # global_params, global_lorenz_func, global_gini, rmse, mafe, max_abs_error = fit_lorenz_curve_decile(
    #     income_shares, lorenz_type, population_shares, error_type='fractional'
    # )

    n_params = int(lorenz_type.split('_')[1])
    print(f"Global Lorenz curve fitted with {n_params} parameters")
    print(f"Global Gini coefficient: {global_gini:.4f}")
    print(f"\nHybrid error statistics (optimized metric):")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAFE: {mafe:.6f}")
    print(f"  Max absolute error: {max_abs_error:.6f}")
    print(f"\nFractional error statistics:")
    print(f"  RMSE: {fractional_rmse:.6f}")
    print(f"  MAFE: {fractional_mafe:.6f}")
    print(f"  Max absolute error: {fractional_max_abs_error:.6f}")
    print(f"\nAbsolute error statistics:")
    print(f"  RMSE: {absolute_rmse:.6f}")
    print(f"  MAFE: {absolute_mafe:.6f}")
    print(f"  Max absolute error: {absolute_max_abs_error:.6f}")

    # Add fitted Lorenz curve values to global_data for comparison
    global_data['income_fraction_fitted'] = global_lorenz_func(
        global_data['population_fraction'].values, *global_params
    )

    # Create a separate DataFrame with just the 100 bins used for fitting
    # This allows direct comparison of fitted vs actual at the fitting points
    p_fitted = np.cumsum(np.concatenate([[0], population_shares]))
    L_actual = np.cumsum(np.concatenate([[0], income_shares]))
    L_fitted = global_lorenz_func(p_fitted, *global_params)

    fitted_bins_data = pd.DataFrame({
        'bin_number': range(len(income_shares)),
        'population_share': population_shares,
        'population_cumulative': p_fitted[1:],
        'income_share_actual': income_shares,
        'income_cumulative_actual': L_actual[1:],
        'income_share_fitted': np.diff(L_fitted),
        'income_cumulative_fitted': L_fitted[1:]
    })

    return (global_params, global_lorenz_func, global_gini,
            rmse, mafe, max_abs_error,
            fractional_rmse, fractional_mafe, fractional_max_abs_error,
            absolute_rmse, absolute_mafe, absolute_max_abs_error,
            global_data, fitted_bins_data)


def compute_global_poverty_metrics(country_results, lorenz_type, poverty_lines):
    """
    Compute global poverty metrics at various poverty lines.

    Parameters:
    -----------
    country_results : pandas DataFrame
        Results from fit_country_lorenz_curves
    lorenz_type : str
        Lorenz curve type: 'pareto_1', 'ortega_2', 'gq_3', 'beta_3', or 'sarabia_3'
    poverty_lines : array_like
        Poverty line thresholds (in PPP dollars/year)

    Returns:
    --------
    metrics : pandas DataFrame
        Poverty headcount and other metrics for each poverty line
    """
    poverty_lines = np.asarray(poverty_lines)
    total_population = country_results['population'].sum()

    results = []

    for poverty_line in poverty_lines:
        people_below = 0

        for idx, row in country_results.iterrows():
            country_mean = row['gdp'] / row['population']
            threshold_fraction = poverty_line / country_mean

            try:
                pop_fractions = evaluate_country_lorenz(row, [threshold_fraction], lorenz_type)
                people_below += pop_fractions[0] * row['population']
            except:
                continue

        results.append({
            'poverty_line': poverty_line,
            'people_below': people_below,
            'poverty_rate': people_below / total_population
        })

    return pd.DataFrame(results)
