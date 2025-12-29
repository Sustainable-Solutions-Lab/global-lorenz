"""
Calculate Gini coefficient from empirical global distribution.

Uses the trapezoidal rule to integrate the Lorenz curve:
Gini = 1 - 2 * ∫L(p)dp from 0 to 1
"""

import pandas as pd
import numpy as np

# Read the empirical global distribution
print("Reading empirical distribution...")
df = pd.read_csv('data/output/empirical_distribution_global.csv')

print(f"Loaded {len(df)} data points\n")

# Extract population and income fractions
p = df['population_fraction'].values
L = df['income_fraction'].values

print("Population fraction range:", p.min(), "to", p.max())
print("Income fraction range:", L.min(), "to", L.max())

# Add origin point (0, 0) if not present
if p[0] > 0:
    p = np.concatenate([[0], p])
    L = np.concatenate([[0], L])
    print("\nAdded origin point (0, 0)")

# Calculate area under Lorenz curve using trapezoidal rule
area_under_lorenz = np.trapezoid(L, p)
print(f"\nArea under Lorenz curve: {area_under_lorenz:.6f}")

# Calculate Gini coefficient
# Gini = 1 - 2 * integral(L(p))
# This is because the area between the line of equality and Lorenz curve
# is (0.5 - area_under_lorenz), and Gini = 2 * (0.5 - area_under_lorenz)
gini = 1 - 2 * area_under_lorenz

print(f"\n{'='*60}")
print(f"EMPIRICAL GINI COEFFICIENT: {gini:.6f}")
print(f"{'='*60}")

# Alternative calculation for verification
# Gini can also be calculated as: A / (A + B)
# where A = area between line of equality and Lorenz curve
#       B = area under Lorenz curve
area_A = 0.5 - area_under_lorenz  # Area between equality line and Lorenz curve
area_B = area_under_lorenz         # Area under Lorenz curve
gini_alt = area_A / (area_A + area_B)

print(f"\nVerification (alternative formula): {gini_alt:.6f}")
print(f"Difference: {abs(gini - gini_alt):.10f}")

# Show comparison with perfect equality and perfect inequality
print(f"\nComparison:")
print(f"  Perfect equality (Gini = 0.0): Everyone has same income")
print(f"  Your data (Gini = {gini:.4f}): Current global distribution")
print(f"  Perfect inequality (Gini = 1.0): One person has all income")

# Calculate income concentration
print(f"\nIncome concentration from data:")
# Top 10% share
if len(df[df['population_fraction'] >= 0.9]) > 0:
    top10_L = df[df['population_fraction'] >= 0.9].iloc[0]['income_fraction']
    top10_share = 1 - top10_L
    print(f"  Top 10% earn: {top10_share*100:.2f}% of global income")

# Top 1% share
if len(df[df['population_fraction'] >= 0.99]) > 0:
    top1_L = df[df['population_fraction'] >= 0.99].iloc[0]['income_fraction']
    top1_share = 1 - top1_L
    print(f"  Top 1% earn: {top1_share*100:.2f}% of global income")

# Bottom 50% share
if len(df[df['population_fraction'] >= 0.5]) > 0:
    bottom50_L = df[df['population_fraction'] >= 0.5].iloc[0]['income_fraction']
    print(f"  Bottom 50% earn: {bottom50_L*100:.2f}% of global income")
