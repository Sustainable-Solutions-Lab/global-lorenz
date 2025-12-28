"""
Main workflow script for fitting global Lorenz curves.

This script demonstrates the complete workflow:
1. Load country-level income data from Excel
2. Fit Lorenz curves at country level (with 1, 2, or 3 parameters)
3. Aggregate to global income distribution
4. Fit global Lorenz curve
5. Generate visualizations and reports
"""

import sys
import traceback
from datetime import datetime
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
from global_lorenz.country_fitting import read_country_data, filter_most_recent_complete, prepare_lorenz_data
from global_lorenz.global_aggregation import global_distribution_to_lorenz


def plot_lorenz_curves(country_results, lorenz_type, output_dir):
    """
    Plot sample country Lorenz curves.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

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

    n_params = int(lorenz_type.split('_')[1])

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


def plot_global_lorenz(p, L, params, lorenz_type, output_dir):
    """
    Plot global Lorenz curve.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

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


def run_workflow(input_file, income_cols, lorenz_type):
    """
    Run the complete workflow.

    Parameters:
    -----------
    input_file : str
        Path to Excel file with country data
    income_cols : list
        Column names for income distribution data
    lorenz_type : str
        Type of Lorenz curve function (function name without 'lorenz_' prefix).
        Options:
        - 'pareto_1': 1-parameter Pareto Lorenz
        - 'ortega_2': 2-parameter Ortega/Jantzen-Volpert
        - 'gq_3': 3-parameter generalized quadratic
        - 'beta_3': 3-parameter Kakwani beta
        - 'sarabia_3': 3-parameter Sarabia ordered family
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path('data/output') / f"{lorenz_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"\n2. Fitting {lorenz_type} Lorenz curves at country level...")
    country_results = fit_country_lorenz_curves(
        data_df,
        income_cols,
        lorenz_type,
    )
    print(f"   Successfully fitted {len(country_results)} countries")
    print(f"   Mean Gini: {country_results['gini'].mean():.3f}")
    print(f"   Median Gini: {country_results['gini'].median():.3f}")
    
    # Save country results
    country_results.to_csv(output_dir / f'country_results_{lorenz_type}.csv', index=False)
    print(f"   Saved results to {output_dir / f'country_results_{lorenz_type}.csv'}")

    # Step 3: Plot country curves
    print("\n3. Plotting sample country Lorenz curves...")
    plot_lorenz_curves(country_results, lorenz_type, output_dir)
    
    # Step 4: Aggregate to global distribution
    print(f"\n4. Aggregating to global distribution...")
    global_params, global_lorenz_func, global_gini, global_data = fit_global_lorenz(
        country_results,
        lorenz_type,
        None,
    )

    # Save global data
    global_data.to_csv(output_dir / 'global_distribution.csv', index=False)
    print(f"   Saved global distribution to {output_dir / 'global_distribution.csv'}")

    # Step 5: Plot global Lorenz curve
    print("\n5. Plotting global Lorenz curve...")
    p, L = global_distribution_to_lorenz(global_data)
    plot_global_lorenz(p, L, global_params, lorenz_type, output_dir)
    
    # Step 6: Generate summary report
    print("\n6. Generating summary report...")
    report = []
    report.append("=" * 70)
    report.append("GLOBAL LORENZ CURVE FITTING SUMMARY")
    report.append("=" * 70)
    report.append(f"\nLorenz curve type: {lorenz_type}")
    report.append(f"Number of countries: {len(country_results)}")
    report.append(f"Mean country Gini: {country_results['gini'].mean():.4f}")
    report.append(f"Median country Gini: {country_results['gini'].median():.4f}")
    report.append(f"\nGlobal Gini coefficient: {global_gini:.4f}")
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
    
    for lorenz_type in ['pareto_1', 'ortega_2', 'gq_3', 'beta_3', 'sarabia_3']:
        print(f"\n\nConfiguration: {lorenz_type}")
        print("-" * 70)

        try:
            run_workflow(
                input_file,
                income_cols,
                lorenz_type,
            )
        except Exception as e:
            print(f"Error with configuration {lorenz_type}: {e}")
            traceback.print_exc()
