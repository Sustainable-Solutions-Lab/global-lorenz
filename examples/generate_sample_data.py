"""
Generate sample country-level income data for testing.

This script creates a synthetic dataset with realistic income distributions,
GDP, and population data for demonstration purposes.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_lorenz_data(n_countries=50, random_seed=42):
    """
    Generate synthetic country-level income distribution data.
    
    Parameters:
    -----------
    n_countries : int
        Number of countries to generate
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    df : pandas DataFrame
        Country-level data with income shares, GDP, and population
    """
    np.random.seed(random_seed)
    
    data = []
    
    for i in range(n_countries):
        country_name = f"Country_{i+1:02d}"
        
        # Generate income distribution with varying inequality
        # Using a Pareto-like distribution for realism
        gini_target = np.random.uniform(0.25, 0.60)
        
        # Generate income shares for deciles
        # Use exponential distribution with varying rate
        rate = np.random.uniform(0.5, 2.5)
        shares = np.exp(-rate * np.linspace(0, 1, 10))
        shares = shares[::-1]  # Reverse so poorest has least
        shares = shares / shares.sum()  # Normalize
        
        # Adjust to match target Gini (approximate)
        adjustment = (gini_target - 0.35) / 0.2
        shares = shares ** (1 + 0.5 * adjustment)
        shares = shares / shares.sum()
        
        # Generate GDP and population
        # GDP: log-normal distribution (realistic for countries)
        log_gdp = np.random.normal(24, 2)  # Mean around $26B
        gdp = np.exp(log_gdp)
        
        # Population: log-normal distribution
        log_pop = np.random.normal(16, 1.5)  # Mean around 9M
        population = np.exp(log_pop)
        
        row = {
            'Country': country_name,
            'GDP': gdp,
            'Population': population,
        }
        
        # Add income shares
        for j, share in enumerate(shares):
            row[f'D{j+1}'] = share
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def generate_realistic_data(n_countries=30, random_seed=42):
    """
    Generate more realistic country data with named countries.
    """
    np.random.seed(random_seed)
    
    # Sample country names
    countries = [
        "United States", "China", "India", "Brazil", "Japan",
        "Germany", "United Kingdom", "France", "Italy", "Canada",
        "South Korea", "Spain", "Mexico", "Indonesia", "Netherlands",
        "Saudi Arabia", "Turkey", "Switzerland", "Poland", "Belgium",
        "Sweden", "Argentina", "Norway", "Austria", "Ireland",
        "Israel", "Denmark", "Singapore", "Malaysia", "Philippines",
    ]
    
    # Realistic inequality patterns (Gini coefficients)
    gini_values = [
        0.41, 0.38, 0.47, 0.53, 0.33,  # US, China, India, Brazil, Japan
        0.32, 0.35, 0.32, 0.36, 0.33,  # Germany, UK, France, Italy, Canada
        0.35, 0.36, 0.46, 0.39, 0.28,  # Korea, Spain, Mexico, Indonesia, Netherlands
        0.42, 0.43, 0.33, 0.30, 0.27,  # Saudi, Turkey, Switzerland, Poland, Belgium
        0.28, 0.41, 0.27, 0.30, 0.32,  # Sweden, Argentina, Norway, Austria, Ireland
        0.39, 0.28, 0.46, 0.41, 0.44,  # Israel, Denmark, Singapore, Malaysia, Philippines
    ]
    
    # Realistic GDP values (billions, PPP)
    gdp_values = [
        21e12, 25e12, 9e12, 3.5e12, 5.5e12,
        4.5e12, 3.1e12, 3e12, 2.5e12, 1.9e12,
        2.3e12, 1.8e12, 2.6e12, 3.2e12, 1.0e12,
        1.8e12, 2.4e12, 0.7e12, 1.3e12, 0.6e12,
        0.55e12, 0.9e12, 0.4e12, 0.5e12, 0.5e12,
        0.4e12, 0.35e12, 0.6e12, 0.9e12, 1.0e12,
    ]
    
    # Realistic population values
    pop_values = [
        330e6, 1400e6, 1380e6, 212e6, 126e6,
        83e6, 67e6, 67e6, 60e6, 38e6,
        52e6, 47e6, 128e6, 270e6, 17e6,
        34e6, 84e6, 8.6e6, 38e6, 11.5e6,
        10.3e6, 45e6, 5.4e6, 8.9e6, 4.9e6,
        9.2e6, 5.8e6, 5.7e6, 32e6, 109e6,
    ]
    
    data = []
    
    for i in range(n_countries):
        country = countries[i]
        gini = gini_values[i]
        gdp = gdp_values[i]
        pop = pop_values[i]
        
        # Generate income shares that approximately match Gini
        # Use power law with adjustment
        alpha = 1.0 + (gini - 0.35) * 3
        shares = np.array([(j+1)**alpha for j in range(10)])
        shares = shares / shares.sum()
        
        row = {
            'Country': country,
            'GDP': gdp,
            'Population': pop,
        }
        
        for j, share in enumerate(shares):
            row[f'D{j+1}'] = share
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    # Generate synthetic data
    print("Generating synthetic country data...")
    df_synthetic = generate_lorenz_data(n_countries=50)
    output_file = output_dir / 'synthetic_country_data.xlsx'
    df_synthetic.to_excel(output_file, index=False)
    print(f"Saved synthetic data to {output_file}")
    print(f"  {len(df_synthetic)} countries")
    print(f"  Total population: {df_synthetic['Population'].sum()/1e9:.2f} billion")
    print(f"  Total GDP: ${df_synthetic['GDP'].sum()/1e12:.2f} trillion")
    
    # Generate realistic data
    print("\nGenerating realistic country data...")
    df_realistic = generate_realistic_data(n_countries=30)
    output_file = output_dir / 'realistic_country_data.xlsx'
    df_realistic.to_excel(output_file, index=False)
    print(f"Saved realistic data to {output_file}")
    print(f"  {len(df_realistic)} countries")
    print(f"  Total population: {df_realistic['Population'].sum()/1e9:.2f} billion")
    print(f"  Total GDP: ${df_realistic['GDP'].sum()/1e12:.2f} trillion")
    
    print("\nExample data preview:")
    print(df_realistic.head())
