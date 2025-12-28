"""
Main workflow script for fitting global Lorenz curves.

This script demonstrates the complete workflow:
1. Load country-level income data from Excel
2. Fit Lorenz curves at country level (with 1, 2, or 3 parameters)
3. Aggregate to global income distribution
4. Fit global Lorenz curve
5. Generate visualizations and reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from global_lorenz import (
    fit_country_lorenz_curves,
    aggregate_global_distribution,
    fit_global_lorenz,
    lorenz_pareto_1,
    lorenz_ortega_2,
    lorenz_gq_3,
)
from global_lorenz.country_fitting import read_country_data, filter_most_recent_complete


def plot_lorenz_curves(country_results, n_params, output_dir='output'):
    """
    Plot sample country Lorenz curves.
    """
    from global_lorenz.country_fitting import prepare_lorenz_data

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Select appropriate function
    if n_params == 1:
        lorenz_func = lorenz_pareto_1
    elif n_params == 2:
        lorenz_func = lorenz_ortega_2
    else:
        lorenz_func = lorenz_gq_3

    # Plot a few example countries
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (idx, row) in enumerate(country_results.head(4).iterrows()):
        ax = axes[i]

        # Get parameters
        params = tuple(row[f'param_{j+1}'] for j in range(n_params))

        # Plot fitted curve
        p = np.linspace(0, 1, 100)
        L = lorenz_func(p, *params)
        
        ax.plot(p, L, 'b-', label='Fitted Lorenz curve', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect equality', alpha=0.5)
        
        ax.set_xlabel('Cumulative population fraction')
        ax.set_ylabel('Cumulative income fraction')
        ax.set_title(f"{row['country']}\nGini = {row['gini']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'country_lorenz_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved country Lorenz curves to {output_dir / 'country_lorenz_curves.png'}")


def plot_global_lorenz(p, L, params, n_params, output_dir='output'):
    """
    Plot global Lorenz curve.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Select appropriate function
    if n_params == 1:
        lorenz_func = lorenz_pareto_1
    elif n_params == 2:
        lorenz_func = lorenz_ortega_2
    else:
        lorenz_func = lorenz_gq_3

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot data points
    ax.scatter(p[::10], L[::10], c='red', s=30, alpha=0.5, label='Aggregated data', zorder=3)
    
    # Plot fitted curve
    p_smooth = np.linspace(0, 1, 200)
    L_smooth = lorenz_func(p_smooth, *params)
    ax.plot(p_smooth, L_smooth, 'b-', label='Fitted global Lorenz curve', linewidth=2)
    
    # Perfect equality line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect equality', alpha=0.5)
    
    ax.set_xlabel('Cumulative population fraction', fontsize=12)
    ax.set_ylabel('Cumulative income fraction', fontsize=12)
    ax.set_title('Global Lorenz Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'global_lorenz_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved global Lorenz curve to {output_dir / 'global_lorenz_curve.png'}")


def run_workflow(input_file, income_cols,
                 n_params_country=2, n_params_global=2,
                 curve_type_country='quadratic', curve_type_global='quadratic',
                 output_dir='output'):
    """
    Run the complete workflow.

    Parameters:
    -----------
    input_file : str
        Path to Excel file with country data
    income_cols : list
        Column names for income distribution data
    n_params_country : int
        Number of parameters for country-level fits (1, 2, or 3)
    n_params_global : int
        Number of parameters for global fit (1, 2, or 3)
    curve_type_country : str
        For n_params_country=3: 'quadratic', 'beta', or 'sarabia'
    curve_type_global : str
        For n_params_global=3: 'quadratic', 'beta', or 'sarabia'
    output_dir : str
        Directory for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Global Lorenz Curve Fitting Workflow")
    print("=" * 70)

    # Step 1: Load data
    print("\n1. Loading country data...")
    raw_data = read_country_data(input_file)
    print(f"   Loaded {len(raw_data)} total rows")

    # Filter to most recent complete data per country
    data_df = filter_most_recent_complete(raw_data, income_cols)
    print(f"   Filtered to {len(data_df)} countries with most recent complete data")

    # Step 2: Fit country-level Lorenz curves
    curve_desc = f"{n_params_country}-parameter"
    if n_params_country == 3:
        curve_desc += f" ({curve_type_country})"
    print(f"\n2. Fitting {curve_desc} Lorenz curves at country level...")
    country_results = fit_country_lorenz_curves(
        data_df, income_cols,
        n_params=n_params_country,
        curve_type=curve_type_country
    )
    print(f"   Successfully fitted {len(country_results)} countries")
    print(f"   Mean Gini: {country_results['gini'].mean():.3f}")
    print(f"   Median Gini: {country_results['gini'].median():.3f}")
    
    # Save country results
    country_results.to_csv(output_dir / f'country_results_{n_params_country}param.csv', index=False)
    print(f"   Saved results to {output_dir / f'country_results_{n_params_country}param.csv'}")
    
    # Step 3: Plot country curves
    print("\n3. Plotting sample country Lorenz curves...")
    plot_lorenz_curves(country_results, n_params_country, output_dir)
    
    # Step 4: Aggregate to global distribution
    print(f"\n4. Aggregating to global distribution...")
    global_params, global_lorenz_func, global_gini, global_data = fit_global_lorenz(
        country_results,
        n_params_country,
        n_params_global,
    )
    
    # Save global data
    global_data.to_csv(output_dir / 'global_distribution.csv', index=False)
    print(f"   Saved global distribution to {output_dir / 'global_distribution.csv'}")
    
    # Step 5: Plot global Lorenz curve
    print("\n5. Plotting global Lorenz curve...")
    from global_lorenz.global_aggregation import global_distribution_to_lorenz
    p, L = global_distribution_to_lorenz(global_data)
    plot_global_lorenz(p, L, global_params, n_params_global, output_dir)
    
    # Step 6: Generate summary report
    print("\n6. Generating summary report...")
    report = []
    report.append("=" * 70)
    report.append("GLOBAL LORENZ CURVE FITTING SUMMARY")
    report.append("=" * 70)
    report.append(f"\nCountry-level fits: {n_params_country} parameters")
    report.append(f"Number of countries: {len(country_results)}")
    report.append(f"Mean country Gini: {country_results['gini'].mean():.4f}")
    report.append(f"Median country Gini: {country_results['gini'].median():.4f}")
    report.append(f"\nGlobal fit: {n_params_global} parameters")
    report.append(f"Global Gini coefficient: {global_gini:.4f}")
    report.append(f"\nGlobal Lorenz curve parameters:")
    for i, param in enumerate(global_params):
        report.append(f"  Parameter {i+1}: {param:.6f}")
    report.append("\n" + "=" * 70)
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\nWorkflow complete! Results saved to {output_dir}/")
    
    return country_results, global_params, global_gini


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file>")
        print("\nExample: python main.py data/input/pip_2025-12-28.xlsx")
        print("  (assumes columns named decile1, decile2, ..., decile10)")
        sys.exit(1)

    input_file = sys.argv[1]

    # Use actual column names from World Bank PIP data
    income_cols = [f'decile{i}' for i in range(1, 11)]
    
    # Run workflow with different parameter configurations
    print("\n" + "=" * 70)
    print("Testing different Lorenz curve forms")
    print("=" * 70)
    
    for n_country, n_global in [(1, 1), (2, 2), (3, 3)]:
        print(f"\n\nConfiguration: {n_country}-param country, {n_global}-param global")
        print("-" * 70)
        
        try:
            run_workflow(
                input_file,
                income_cols,
                n_params_country=n_country,
                n_params_global=n_global,
                output_dir=f'output_{n_country}param'
            )
        except Exception as e:
            print(f"Error with configuration ({n_country}, {n_global}): {e}")
            import traceback
            traceback.print_exc()
