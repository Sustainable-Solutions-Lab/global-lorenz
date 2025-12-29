# Global Lorenz Curve Fitting

Fit global Lorenz curves to World Bank income distribution data.

## Features

- **Country-level Lorenz curve fitting**: Fit Lorenz curves to income distribution data for individual countries using a population-weighted fractional error objective that gives equal weight to all deciles
- **Multiple functional forms**: Support for 5 different Lorenz curve types:
  - 1-parameter: Pareto (`pareto_1`)
  - 2-parameter: Ortega/Jantzen-Volpert (`ortega_2`)
  - 3-parameter: Generalized Quadratic (`gq_3`), Beta (`beta_3`), Sarabia (`sarabia_3`)
- **Goodness-of-fit statistics**: RMSE, mean absolute fractional error, and maximum absolute fractional error for each country
- **Global aggregation**: Aggregate country-level distributions to create a global income distribution
- **Global Lorenz curve**: Fit a Lorenz curve to the aggregated global distribution using the same population-weighted methodology as country-level fitting
- **Unified methodology**: Both country and global fits use the same fractional error objective, ensuring consistency across scales
- **Comprehensive output**: CSV files include actual vs fitted decile shares, year, and fit quality metrics
- **Poverty metrics**: Calculate global poverty headcount at various poverty lines

## Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Usage

### Quick Start

```bash
python main.py data/input/pip_2025-12-28.xlsx
```

This will automatically:
- Read data from World Bank PIP format (Excel or CSV)
- Filter to the most recent year for each country with complete data
- Fit all 5 Lorenz curve types (pareto_1, ortega_2, gq_3, beta_3, sarabia_3)
- Generate output in timestamped directories: `data/output/{lorenz_type}_{timestamp}/`

Each run tests all Lorenz curve types and saves results separately, allowing you to compare their performance.

### Expected Data Format

The input file should be in World Bank Poverty and Inequality Platform (PIP) format with the following columns:

#### Distributional shares (deciles)

**decile1 ... decile10**

Share of total welfare accruing to each decile:
- `decile1` = poorest 10%
- `decile2` = second poorest 10%
- ...
- `decile10` = richest 10%

These values sum to 1.0 for each country-year observation.

#### Population & macro aggregates

**reporting_pop**

Population used to weight this country-year observation.

**reporting_gdp**

GDP associated with the reporting year (PPP-consistent).

**reporting_pce**

Private consumption expenditure (PPP-consistent).

#### Other required columns

**country_name**

Name of the country.

**reporting_year**

Year of the data observation.

### Data Filtering

When the input data contains multiple years for each country, the script automatically selects the most recent year that has complete data (all decile columns, `reporting_pop`, and `reporting_gdp` are non-null)

### Programmatic Usage

```python
from global_lorenz import (
    fit_country_lorenz_curves,
    fit_global_lorenz,
)
from global_lorenz.country_fitting import (
    read_country_data,
    filter_most_recent_complete,
)

# Load data
raw_data = read_country_data('data/input/pip_2025-12-28.xlsx')

# Define income columns
income_cols = [f'decile{i}' for i in range(1, 11)]

# Filter to most recent complete data per country
data_df = filter_most_recent_complete(raw_data, income_cols)

# Fit country-level Lorenz curves (2-parameter form)
country_results = fit_country_lorenz_curves(
    data_df,
    income_cols,
    'ortega_2'
)

# Fit global Lorenz curve
global_params, global_lorenz_func, global_gini, global_data = fit_global_lorenz(
    country_results,
    'ortega_2',
    None
)

print(f"Global Gini coefficient: {global_gini:.4f}")
```

### Using Different Curve Forms

For 3-parameter fits, you can choose between three different forms:

```python
# Fit using Generalized Quadratic (implicit equation)
country_results_gq = fit_country_lorenz_curves(
    data_df,
    income_cols,
    'gq_3'
)

# Fit using Sarabia Ordered Family curve
country_results_sarabia = fit_country_lorenz_curves(
    data_df,
    income_cols,
    'sarabia_3'
)

# Fit using Beta Lorenz curve
country_results_beta = fit_country_lorenz_curves(
    data_df,
    income_cols,
    'beta_3'
)
```

## Lorenz Curve Forms

### 1-Parameter Form: `lorenz_pareto_1` (Pareto Lorenz)

```
L(p) = 1 - (1 - p)^(1 - 1/a)
```

This is the Pareto distribution Lorenz curve. The Gini index is G = 1/(2a - 1).

**Reference:** https://www.mdpi.com/2225-1146/13/3/30

### 2-Parameter Form: `lorenz_ortega_2` (Ortega/Jantzen-Volpert)

```
L(p) = p^a · (1 - (1-p)^b)
```

Constraints: a >= 0, 0 < b <= 1

This is a flexible form commonly used for income distributions.

**References:**
- Ortega, P., M.A. Fernández, M. Ladoux, A. García (1991). A new functional form for estimating Lorenz curves. Review of Income and Wealth, 37, 447-452.

### 3-Parameter Forms

The package supports three different 3-parameter Lorenz curve forms:

#### `lorenz_gq_3`: Generalized Quadratic (selected with `lorenz_type='gq_3'`)

Solves the implicit equation:
```
L(1-L) = a(p² - L) + bL(p-1) + c(p-L)
```

This reduces to a quadratic equation in L and selects the solution between 0 and 1.

**References:**
- Villasenor, J., and B. Arnold (1989). Elliptical Lorenz curves. Journal of Econometrics, 40, 327–338.
- https://rpubs.com/tsamuel/709170

#### `lorenz_sarabia_3`: Sarabia Ordered Family (selected with `lorenz_type='sarabia_3'`)

```
L(p) = p^a · (1 - (1-p)^b)^c
```

With constraints: a > 0, b > 0, c > 1

**Reference:** Sarabia, J., E. Castillo and D. Slottje (1999). An Ordered Family of Lorenz Curves, Journal of Econometrics, 91, 43-60.

#### `lorenz_beta_3`: Beta Lorenz (selected with `lorenz_type='beta_3'`)

```
L(p) = p - a · p^b · (1-p)^c
```

With constraints: 0 < a, b, c < 1

**Reference:** Kakwani, N. (1980). On a Class of Poverty Measures, Econometrica, 48, 437–446.

## Output

The workflow generates timestamped output directories (e.g., `data/output/ortega_2_20251228-140106/`) containing:

### Files Generated

1. **`country_results_{lorenz_type}.csv`**: Country-level fit results with columns:
   - `country`: Country name
   - `year`: Reporting year for the data
   - `gini`: Gini coefficient
   - `rmse`: Root mean squared fractional error
   - `mafe`: Mean absolute fractional error
   - `max_abs_error`: Maximum absolute fractional error across deciles
   - `param_1`, `param_2`, `param_3`: Fitted Lorenz curve parameters
   - `gdp`, `population`: Country economic data
   - `decile1_actual` ... `decile10_actual`: Input income shares
   - `decile1_fitted` ... `decile10_fitted`: Predicted income shares from fitted curve

2. **`global_distribution_{lorenz_type}.csv`**: Global income distribution data

3. **`country_lorenz_curves_{lorenz_type}.png`**: Sample country Lorenz curve plots

4. **`global_lorenz_curve_{lorenz_type}.png`**: Fitted global Lorenz curve plot

5. **`summary_report_{lorenz_type}.txt`**: Summary statistics including:
   - Country-level Gini statistics (mean, median)
   - Country-level RMSE statistics (mean, std, min, max)
   - Global Gini coefficient
   - Fitted parameter values

### Interpreting Goodness-of-Fit Statistics

All error metrics are **fractional (relative) errors**, not absolute errors:

- **RMSE = 0.05**: ~5% root mean squared fractional error across deciles
- **MAFE = 0.03**: On average, predicted decile shares differ from actual by 3%
- **max_abs_error = 0.15**: The worst-fit decile has 15% relative error

**Example:** If a decile has actual share = 0.05 and predicted share = 0.055:
- Fractional error = (0.055/0.05) - 1 = 0.10 (10% error)

This fractional error objective gives **equal weight to all deciles** - fitting the small first decile (e.g., 2% of income) just as well as the large tenth decile (e.g., 30% of income).

## Theory

### Lorenz Curves

A Lorenz curve L(p) represents the cumulative proportion of income held by the bottom p proportion of the population. Key properties:

- L(0) = 0: The bottom 0% has 0% of income
- L(1) = 1: The bottom 100% has 100% of income
- L is concave: Income inequality means the curve lies below the diagonal
- Gini coefficient = 1 - 2∫L(p)dp from 0 to 1

### Unified Fitting Methodology

Both country-level and global-level Lorenz curves are fitted using the same **population-weighted fractional error objective**:

```
Minimize: Σᵢ wᵢ · ((predicted_shareᵢ / actual_shareᵢ) - 1)²
```

Where:
- `wᵢ` = population weight for bin i
- `predicted_shareᵢ` = L(pᵢ₊₁) - L(pᵢ) from fitted curve
- `actual_shareᵢ` = observed income share in bin i

**Country-level fitting:**
- Each decile has equal population weight (wᵢ = 0.1)
- This gives equal importance to fitting all deciles, regardless of their absolute income size
- Prevents the optimizer from only fitting large deciles well while ignoring small ones

**Global-level fitting:**
- Population weights vary by bin (wᵢ = pᵢ₊₁ - pᵢ)
- Bins with more people receive higher weight in the objective
- Ensures the global fit accurately represents where most of the world's population lives

This unified approach ensures methodological consistency across scales while appropriately accounting for population distribution.

### Global Aggregation

The global distribution is computed by:

1. For each income threshold y
2. For each country with mean income μ_c
3. Calculate the fraction of country population with income < y using the country Lorenz curve
4. Sum across all countries weighted by population
5. Result: global cumulative distribution function
6. Fit global Lorenz curve to this distribution using population-weighted objective

### Applications

- **Poverty analysis**: Calculate global poverty headcount
- **Inequality measurement**: Compute global Gini coefficient
- **Income distribution**: Understand global income inequality
- **Policy evaluation**: Assess impact of policies on global distribution

## Examples

See `examples/` directory for:
- Sample data generation
- Visualization examples
- Advanced usage patterns

## References

- Lorenz, M. O. (1905). Methods of measuring the concentration of wealth. *Publications of the American Statistical Association*, 9(70), 209-219.
- Gastwirth, J. L. (1972). The estimation of the Lorenz curve and Gini index. *The Review of Economics and Statistics*, 306-316.
- Villasenor, J., and B. Arnold (1989). Elliptical Lorenz curves. *Journal of Econometrics*, 40, 327–338.
- Ortega, P., M.A. Fernández, M. Ladoux, A. García (1991). A new functional form for estimating Lorenz curves. *Review of Income and Wealth*, 37, 447-452.
- Sarabia, J., E. Castillo and D. Slottje (1999). An Ordered Family of Lorenz Curves. *Journal of Econometrics*, 91, 43-60.
- Kakwani, N. (1980). On a Class of Poverty Measures. *Econometrica*, 48, 437–446.
- Hajargasht, G. & Griffiths, W. E. (2022). Pareto-Lorenz Curves (survey of Lorenz curve forms including generalized quadratic and beta). https://fbe.unimelb.edu.au/__data/assets/pdf_file/0006/1965867/2022HajargashtGrifffiths.pdf
- World Bank. (2023). World Development Indicators.

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
