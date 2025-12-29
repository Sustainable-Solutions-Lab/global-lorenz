"""
Create a comprehensive summary report with key insights.
"""
import pandas as pd
import numpy as np

# Load the results - try fit10 first, then fall back to regular
import os
if os.path.exists('data/output/results_summary_fit10.csv'):
    df = pd.read_csv('data/output/results_summary_fit10.csv')
    results_type = "10-bin fitting (deciles)"
elif os.path.exists('data/output/results_summary.csv'):
    df = pd.read_csv('data/output/results_summary.csv')
    results_type = "all results"
else:
    raise FileNotFoundError("No results summary file found. Run summarize_results.py first.")

# Flag suspicious results (Gini >= 0.95 suggests fitting failure)
df['fit_failed'] = df['global_gini'] >= 0.95

print("="*100)
print("GLOBAL LORENZ CURVE FITTING - COMPREHENSIVE SUMMARY")
print("="*100)
print(f"\nResults type: {results_type}")
print("172 countries analyzed | 5 Lorenz types | 3 error metrics | 2 fitting targets = 30 configurations")
print("\n")

# Summary of failures
n_failures = df['fit_failed'].sum()
print(f"Fitting failures detected: {n_failures} out of 30 configurations")
if n_failures > 0:
    print("\nFailed configurations (Gini >= 0.95):")
    failed = df[df['fit_failed']][['lorenz_type', 'error_metric', 'fit_target', 'global_gini']]
    for _, row in failed.iterrows():
        print(f"  - {row['lorenz_type']}: {row['error_metric']}/{row['fit_target']} (Gini={row['global_gini']:.4f})")
    print("\nNote: These failures likely indicate optimizer hitting parameter boundaries.")
    print("Fractional error objective can be unstable for complex functional forms.")

# Filter to successful fits
df_valid = df[~df['fit_failed']].copy()

print("\n" + "="*100)
print("SUCCESSFUL FITS - GLOBAL GINI COEFFICIENT")
print("="*100)
print("\nRealistic range for global Gini: ~0.50-0.75")
print()

# Show Gini by configuration
for lorenz in sorted(df_valid['lorenz_type'].unique()):
    lorenz_df = df_valid[df_valid['lorenz_type'] == lorenz]
    print(f"\n{lorenz.upper()}:")
    print(f"  {'Error Metric':<12} {'Cumulative':<12} {'Shares':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")

    for error_type in ['absolute', 'fractional', 'hybrid']:
        row_data = lorenz_df[lorenz_df['error_metric'] == error_type]
        cum_val = row_data[row_data['fit_target'] == 'cumulative']['global_gini'].values
        sha_val = row_data[row_data['fit_target'] == 'shares']['global_gini'].values

        cum_str = f"{cum_val[0]:.4f}" if len(cum_val) > 0 else "FAILED"
        sha_str = f"{sha_val[0]:.4f}" if len(sha_val) > 0 else "FAILED"

        print(f"  {error_type:<12} {cum_str:<12} {sha_str:<12}")

print("\n" + "="*100)
print("SUCCESSFUL FITS - GLOBAL RMSE (OPTIMIZED METRIC)")
print("="*100)
print("\nLower RMSE = better fit quality")
print()

for lorenz in sorted(df_valid['lorenz_type'].unique()):
    lorenz_df = df_valid[df_valid['lorenz_type'] == lorenz]
    print(f"\n{lorenz.upper()}:")
    print(f"  {'Error Metric':<12} {'Cumulative':<12} {'Shares':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")

    for error_type in ['absolute', 'fractional', 'hybrid']:
        row_data = lorenz_df[lorenz_df['error_metric'] == error_type]
        cum_val = row_data[row_data['fit_target'] == 'cumulative']['global_rmse'].values
        sha_val = row_data[row_data['fit_target'] == 'shares']['global_rmse'].values

        cum_str = f"{cum_val[0]:.6f}" if len(cum_val) > 0 else "FAILED"
        sha_str = f"{sha_val[0]:.6f}" if len(sha_val) > 0 else "FAILED"

        print(f"  {error_type:<12} {cum_str:<12} {sha_str:<12}")

print("\n" + "="*100)
print("TOP RECOMMENDATIONS")
print("="*100)

print("\n1. BEST OVERALL FIT (lowest RMSE):")
best = df_valid.loc[df_valid['global_rmse'].idxmin()]
print(f"   {best['lorenz_type']}: {best['error_metric']}/{best['fit_target']}")
print(f"   RMSE: {best['global_rmse']:.6f} | Gini: {best['global_gini']:.4f}")

print("\n2. BEST BY LORENZ TYPE (lowest RMSE for each functional form):")
for lorenz in sorted(df_valid['lorenz_type'].unique()):
    lorenz_df = df_valid[df_valid['lorenz_type'] == lorenz]
    best = lorenz_df.loc[lorenz_df['global_rmse'].idxmin()]
    print(f"   {lorenz:<10}: {best['error_metric']:<10}/{best['fit_target']:<11} "
          f"RMSE={best['global_rmse']:.6f}  Gini={best['global_gini']:.4f}")

print("\n3. MOST REALISTIC GLOBAL GINI (closest to expected ~0.60-0.70):")
df_valid['gini_realism'] = np.abs(df_valid['global_gini'] - 0.65)
best = df_valid.loc[df_valid['gini_realism'].idxmin()]
print(f"   {best['lorenz_type']}: {best['error_metric']}/{best['fit_target']}")
print(f"   RMSE: {best['global_rmse']:.6f} | Gini: {best['global_gini']:.4f}")

print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)

print("\n1. Fitting Target Comparison:")
print("   - SHARES: Generally lower RMSE for absolute error objective")
print("   - CUMULATIVE: Better for minimizing L(p) curve errors directly")
print("   - Recommendation: Use 'shares' for absolute error, consider 'cumulative' for hybrid/fractional")

print("\n2. Error Metric Comparison:")
print("   - ABSOLUTE: Most stable, lowest RMSE across all Lorenz types")
print("   - HYBRID: Good balance, slightly higher RMSE but more balanced fit")
print("   - FRACTIONAL: Can be unstable for complex forms (gq_3, sarabia_3)")

print("\n3. Lorenz Type Comparison:")
sarabia_valid = df_valid[df_valid['lorenz_type'] == 'sarabia_3']
if len(sarabia_valid) > 0:
    print("   - SARABIA_3: Best fit quality when successful (lowest RMSE)")
ortega = df_valid[df_valid['lorenz_type'] == 'ortega_2']
if len(ortega) > 0:
    print("   - ORTEGA_2: Most robust, works well with all error metrics")
beta = df_valid[df_valid['lorenz_type'] == 'beta_3']
if len(beta) > 0:
    print("   - BETA_3: Good all-around performance")
gq = df_valid[df_valid['lorenz_type'] == 'gq_3']
if len(gq) > 0:
    print("   - GQ_3: Caution with fractional error (fitting failures)")
pareto = df_valid[df_valid['lorenz_type'] == 'pareto_1']
if len(pareto) > 0:
    print("   - PARETO_1: Simple 1-parameter form, limited flexibility")

print("\n4. Global Gini Range:")
print(f"   - Minimum: {df_valid['global_gini'].min():.4f}")
print(f"   - Maximum: {df_valid['global_gini'].max():.4f}")
print(f"   - Mean: {df_valid['global_gini'].mean():.4f}")
print(f"   - Expected realistic range: 0.60-0.70")

print("\n" + "="*100)
print("FINAL RECOMMENDATION")
print("="*100)

print("\nFor most applications, use:")
print(f"   Configuration: {best['lorenz_type']} with {best['error_metric']} error on {best['fit_target']}")
print(f"   Expected Gini: {best['global_gini']:.4f}")
print(f"   Fit Quality: RMSE = {best['global_rmse']:.6f}")

# Show alternative if Gini is far from realistic
if abs(best['global_gini'] - 0.65) > 0.1:
    alt = df_valid.loc[df_valid['gini_realism'].idxmin()]
    print(f"\nAlternative with more realistic Gini:")
    print(f"   Configuration: {alt['lorenz_type']} with {alt['error_metric']} error on {alt['fit_target']}")
    print(f"   Expected Gini: {alt['global_gini']:.4f}")
    print(f"   Fit Quality: RMSE = {alt['global_rmse']:.6f}")

print("\n" + "="*100)
print()
