"""
Fit Lorenz curves to empirical global income distribution.

This script takes the empirical distribution (no country-level fitting) and
fits various Lorenz curve types using different objective functions and
fitting strategies.

Comparison: This is the "ground truth" - fitting directly to empirical data
vs. the previous approach of fitting country curves then aggregating.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from global_lorenz.lorenz_curves import fit_lorenz_curve_decile
from global_lorenz.lorenz_curves import lorenz_pareto_1, lorenz_ortega_2, lorenz_gq_3, lorenz_beta_3, lorenz_sarabia_3
from global_lorenz.global_aggregation import resample_to_equal_population_bins


def load_empirical_distribution():
    """Load empirical global distribution from CSV."""
    df = pd.read_csv('data/output/empirical_distribution_global.csv')

    # Extract population and income fractions
    p = df['population_fraction'].values
    L = df['income_fraction'].values

    # Add origin if needed
    if p[0] > 0:
        p = np.concatenate([[0], p])
        L = np.concatenate([[0], L])

    return p, L


def calculate_gini(p, L):
    """Calculate Gini coefficient from Lorenz curve."""
    area_under_lorenz = np.trapezoid(L, p)
    gini = 1 - 2 * area_under_lorenz
    return gini


def fit_single_configuration(income_shares, population_shares, lorenz_type, error_type, fit_on_cumulative):
    """
    Fit a single Lorenz curve configuration.

    Returns:
    --------
    dict with fitted parameters, Gini, and error statistics
    """
    try:
        (params, lorenz_func, gini,
         rmse, mafe, max_abs_error,
         fractional_rmse, fractional_mafe, fractional_max_abs_error,
         absolute_rmse, absolute_mafe, absolute_max_abs_error) = fit_lorenz_curve_decile(
            income_shares,
            lorenz_type,
            population_shares,
            error_type=error_type,
            fit_on_cumulative=fit_on_cumulative
        )

        return {
            'lorenz_type': lorenz_type,
            'error_type': error_type,
            'fit_target': 'cumulative' if fit_on_cumulative else 'shares',
            'success': True,
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
            'params': params,
        }
    except Exception as e:
        print(f"  ERROR: {lorenz_type}/{error_type}/{fit_on_cumulative}: {e}")
        return {
            'lorenz_type': lorenz_type,
            'error_type': error_type,
            'fit_target': 'cumulative' if fit_on_cumulative else 'shares',
            'success': False,
            'error': str(e),
        }


def main():
    print("="*70)
    print("FITTING LORENZ CURVES TO EMPIRICAL DISTRIBUTION")
    print("="*70)

    # Load empirical distribution
    print("\n1. Loading empirical distribution...")
    p, L = load_empirical_distribution()
    print(f"   Loaded {len(p)-1} empirical data points")

    # Calculate empirical Gini
    empirical_gini = calculate_gini(p, L)
    print(f"   Empirical Gini coefficient: {empirical_gini:.6f}")

    # Resample to equal-population bins
    n_bins = 100  # Start with 100 bins (percentiles)
    print(f"\n2. Resampling to {n_bins} equal-population bins...")
    income_shares, population_shares = resample_to_equal_population_bins(p, L, n_bins=n_bins)
    print(f"   Created {len(income_shares)} bins for fitting")

    # Define configurations to test
    lorenz_types = ['pareto_1', 'ortega_2', 'gq_3', 'beta_3', 'sarabia_3']
    error_types = ['hybrid', 'absolute', 'fractional']
    fit_targets = [False, True]  # False = shares, True = cumulative

    total_configs = len(lorenz_types) * len(error_types) * len(fit_targets)
    print(f"\n3. Testing {total_configs} configurations...")
    print(f"   {len(lorenz_types)} Lorenz types × {len(error_types)} error metrics × {len(fit_targets)} fit targets")

    # Fit all configurations
    results = []
    config_num = 0

    for lorenz_type in lorenz_types:
        for error_type in error_types:
            for fit_on_cumulative in fit_targets:
                config_num += 1
                fit_target_str = 'cumulative' if fit_on_cumulative else 'shares'
                print(f"\n   [{config_num}/{total_configs}] {lorenz_type} / {error_type} / {fit_target_str}")

                result = fit_single_configuration(
                    income_shares,
                    population_shares,
                    lorenz_type,
                    error_type,
                    fit_on_cumulative
                )

                if result['success']:
                    print(f"      Gini: {result['gini']:.6f} (vs empirical {empirical_gini:.6f})")
                    print(f"      RMSE: {result['rmse']:.6f}")

                results.append(result)

    # Create results DataFrame
    print("\n4. Compiling results...")
    df_results = pd.DataFrame([r for r in results if r['success']])

    # Add Gini difference from empirical
    df_results['gini_diff'] = abs(df_results['gini'] - empirical_gini)

    # Sort by RMSE
    df_results_sorted = df_results.sort_values('rmse')

    # Save results
    output_dir = Path('data/output')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = output_dir / f'empirical_fit_results_{n_bins}bins_{timestamp}.csv'
    df_results.to_csv(results_file, index=False)
    print(f"   Saved results to: {results_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)

    n_success = len(df_results)
    n_failed = total_configs - n_success
    print(f"\nSuccessful fits: {n_success}/{total_configs}")
    print(f"Failed fits: {n_failed}/{total_configs}")

    if n_failed > 0:
        print("\nFailed configurations:")
        for r in results:
            if not r['success']:
                print(f"  - {r['lorenz_type']}/{r['error_type']}/{r['fit_target']}")

    print("\n" + "="*70)
    print("TOP 10 FITS (by RMSE)")
    print("="*70)
    print("\n{:<12} {:<12} {:<12} {:>10} {:>10} {:>10}".format(
        "Lorenz", "Error", "Target", "RMSE", "Gini", "Gini Diff"
    ))
    print("-"*70)

    for _, row in df_results_sorted.head(10).iterrows():
        print("{:<12} {:<12} {:<12} {:>10.6f} {:>10.6f} {:>10.6f}".format(
            row['lorenz_type'],
            row['error_type'],
            row['fit_target'],
            row['rmse'],
            row['gini'],
            row['gini_diff']
        ))

    print("\n" + "="*70)
    print("BEST FIT BY LORENZ TYPE")
    print("="*70)

    for lorenz_type in lorenz_types:
        subset = df_results[df_results['lorenz_type'] == lorenz_type]
        if len(subset) > 0:
            best = subset.loc[subset['rmse'].idxmin()]
            print(f"\n{lorenz_type}:")
            print(f"  Configuration: {best['error_type']}/{best['fit_target']}")
            print(f"  RMSE: {best['rmse']:.6f}")
            print(f"  Gini: {best['gini']:.6f} (diff: {best['gini_diff']:.6f})")

    print("\n" + "="*70)
    print("COMPARISON: EMPIRICAL vs BEST FIT")
    print("="*70)

    best_overall = df_results_sorted.iloc[0]
    print(f"\nEmpirical Gini: {empirical_gini:.6f}")
    print(f"Best fit Gini:  {best_overall['gini']:.6f}")
    print(f"Difference:     {best_overall['gini_diff']:.6f}")
    print(f"\nBest configuration: {best_overall['lorenz_type']} / {best_overall['error_type']} / {best_overall['fit_target']}")
    print(f"RMSE: {best_overall['rmse']:.6f}")

    print("\n" + "="*70)

    return df_results


if __name__ == '__main__':
    results = main()
