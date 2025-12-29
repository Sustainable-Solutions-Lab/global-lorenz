"""
Summarize results from all Lorenz curve fitting runs.
"""
import os
import re
import pandas as pd
from pathlib import Path

def parse_summary_report(filepath):
    """Extract key metrics from a summary report file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract configuration
    lorenz_match = re.search(r'Lorenz curve type: (\w+)', content)
    lorenz_type = lorenz_match.group(1) if lorenz_match else None

    fit_target_match = re.search(r'Global-level:.*?Fitting target: ([^\n]+)', content, re.DOTALL)
    if fit_target_match:
        fit_target = fit_target_match.group(1).strip()
        # Clean up the fit target text
        fit_target = fit_target.replace('income shares', 'shares').replace('cumulative L(p) values', 'cumulative')
    else:
        fit_target = None

    error_metric_match = re.search(r'Global-level:.*?Error metric: ([^\n]+)', content, re.DOTALL)
    error_metric = error_metric_match.group(1).strip() if error_metric_match else None

    # Extract country-level metrics
    country_gini_match = re.search(r'Country-level Gini statistics:\s+Mean: ([\d.]+)', content)
    country_gini = float(country_gini_match.group(1)) if country_gini_match else None

    # Extract optimized country-level error metric (first one listed)
    country_rmse_match = re.search(r'RMSE - Mean: ([\d.]+)', content)
    country_rmse = float(country_rmse_match.group(1)) if country_rmse_match else None

    country_mafe_match = re.search(r'MAFE - Mean: ([\d.]+)', content)
    country_mafe = float(country_mafe_match.group(1)) if country_mafe_match else None

    country_max_match = re.search(r'Max error - Mean: ([\d.]+)', content)
    country_max = float(country_max_match.group(1)) if country_max_match else None

    # Extract global metrics
    global_gini_match = re.search(r'Global Gini coefficient: ([\d.]+)', content)
    global_gini = float(global_gini_match.group(1)) if global_gini_match else None

    # Extract optimized global error metric (first one listed under "Global goodness-of-fit")
    global_section = re.search(r'Global goodness-of-fit statistics:(.*?)(?=Global Lorenz curve parameters:|$)',
                               content, re.DOTALL)
    if global_section:
        global_text = global_section.group(1)
        global_rmse_match = re.search(r'RMSE: ([\d.]+)', global_text)
        global_rmse = float(global_rmse_match.group(1)) if global_rmse_match else None

        global_mafe_match = re.search(r'MAFE: ([\d.]+)', global_text)
        global_mafe = float(global_mafe_match.group(1)) if global_mafe_match else None

        global_max_match = re.search(r'Max absolute error: ([\d.]+)', global_text)
        global_max = float(global_max_match.group(1)) if global_max_match else None
    else:
        global_rmse = global_mafe = global_max = None

    return {
        'lorenz_type': lorenz_type,
        'fit_target': fit_target,
        'error_metric': error_metric,
        'country_gini_mean': country_gini,
        'country_rmse_mean': country_rmse,
        'country_mafe_mean': country_mafe,
        'country_max_mean': country_max,
        'global_gini': global_gini,
        'global_rmse': global_rmse,
        'global_mafe': global_mafe,
        'global_max': global_max,
    }


def main():
    # Find all summary report files
    output_dir = Path('data/output')
    results = []

    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            # Find summary report in this directory
            summary_files = list(subdir.glob('summary_report_*.txt'))
            if summary_files:
                report_path = summary_files[0]
                print(f"Processing: {subdir.name}")
                metrics = parse_summary_report(report_path)
                metrics['directory'] = subdir.name

                # Extract bin counts from directory name if present
                # Format: {lorenz}_{error}_{target}_agg{N}_fit{M}_{timestamp}
                dir_parts = subdir.name.split('_')
                if 'agg' in subdir.name and 'fit' in subdir.name:
                    for i, part in enumerate(dir_parts):
                        if part.startswith('agg'):
                            metrics['n_agg_bins'] = int(part[3:])
                        elif part.startswith('fit'):
                            metrics['n_fit_bins'] = int(part[3:])
                else:
                    metrics['n_agg_bins'] = None
                    metrics['n_fit_bins'] = None

                results.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Filter to only 10-bin fits if they exist
    if 'n_fit_bins' in df.columns and df['n_fit_bins'].notna().any():
        df_10bins = df[df['n_fit_bins'] == 10].copy()
        if len(df_10bins) > 0:
            print(f"\n\nFiltering to {len(df_10bins)} runs with 10 fitting bins")
            df = df_10bins

    # Sort by lorenz type, error metric, fit target
    df = df.sort_values(['lorenz_type', 'error_metric', 'fit_target'])

    # Create summary table
    print("\n" + "="*100)
    print("GLOBAL LORENZ CURVE FITTING RESULTS SUMMARY")
    print("="*100)

    # Check if we're showing 10-bin results
    if 'n_fit_bins' in df.columns and df['n_fit_bins'].notna().all() and (df['n_fit_bins'] == 10).all():
        n_fit_bins = 10
        print(f"\nResults for 10-bin fitting (deciles)")
    else:
        n_fit_bins = None

    print(f"\n{len(df)} configurations (5 Lorenz types × 3 error metrics × 2 fitting targets)")
    print("\n")

    # Display columns in a nice format
    display_df = df[[
        'lorenz_type', 'error_metric', 'fit_target',
        'country_gini_mean', 'global_gini',
        'global_rmse', 'global_mafe', 'global_max'
    ]].copy()

    display_df.columns = [
        'Lorenz', 'Error Type', 'Fit Target',
        'Country Gini', 'Global Gini',
        'Global RMSE', 'Global MAFE', 'Global Max Err'
    ]

    # Format numbers
    for col in ['Country Gini', 'Global Gini', 'Global RMSE', 'Global MAFE', 'Global Max Err']:
        display_df[col] = display_df[col].apply(lambda x: f'{x:.6f}' if pd.notna(x) else 'N/A')

    print(display_df.to_string(index=False))

    # Save to CSV
    if n_fit_bins == 10:
        output_file = 'data/output/results_summary_fit10.csv'
    else:
        output_file = 'data/output/results_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")

    # Create pivot tables for easier comparison
    print("\n" + "="*100)
    print("GLOBAL GINI COEFFICIENT BY CONFIGURATION")
    print("="*100)
    for lorenz in df['lorenz_type'].unique():
        lorenz_df = df[df['lorenz_type'] == lorenz]
        pivot = lorenz_df.pivot(index='error_metric', columns='fit_target', values='global_gini')
        print(f"\n{lorenz.upper()}:")
        print(pivot.to_string())

    print("\n" + "="*100)
    print("GLOBAL RMSE (OPTIMIZED METRIC) BY CONFIGURATION")
    print("="*100)
    for lorenz in df['lorenz_type'].unique():
        lorenz_df = df[df['lorenz_type'] == lorenz]
        pivot = lorenz_df.pivot(index='error_metric', columns='fit_target', values='global_rmse')
        print(f"\n{lorenz.upper()}:")
        print(pivot.to_string())

    # Find best configurations
    print("\n" + "="*100)
    print("BEST CONFIGURATIONS")
    print("="*100)

    print("\nLowest Global RMSE (by Lorenz type):")
    for lorenz in df['lorenz_type'].unique():
        lorenz_df = df[df['lorenz_type'] == lorenz]
        best = lorenz_df.loc[lorenz_df['global_rmse'].idxmin()]
        print(f"  {lorenz}: {best['error_metric']}/{best['fit_target']} "
              f"(RMSE={best['global_rmse']:.6f}, Gini={best['global_gini']:.4f})")

    print("\nOverall best (lowest Global RMSE):")
    best = df.loc[df['global_rmse'].idxmin()]
    print(f"  {best['lorenz_type']}: {best['error_metric']}/{best['fit_target']} "
          f"(RMSE={best['global_rmse']:.6f}, Gini={best['global_gini']:.4f})")

    print("\n")

if __name__ == '__main__':
    main()
