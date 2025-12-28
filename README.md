# Global Lorenz Curve Fitting

Fit global Lorenz curves to World Bank income distribution data.

## Features

- **Country-level Lorenz curve fitting**: Fit Lorenz curves to income distribution data for individual countries
- **Multiple functional forms**: Support for 1, 2, or 3 parameter Lorenz curves
- **Global aggregation**: Aggregate country-level distributions to create a global income distribution
- **Global Lorenz curve**: Fit a Lorenz curve to the aggregated global distribution
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

Or with the CSV version:

```bash
python main.py data/input/pip.csv
```

The script automatically:
- Reads data from World Bank PIP format (Excel or CSV)
- Filters to the most recent year for each country with complete data
- Fits Lorenz curves at country and global levels

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
    n_params=2
)

# Fit global Lorenz curve
global_params, global_lorenz_func, global_gini, global_data = fit_global_lorenz(
    country_results,
    n_params_country=2,
    n_params_global=2
)

print(f"Global Gini coefficient: {global_gini:.4f}")
```

### Using Different Curve Forms

For 3-parameter fits, you can choose between three different forms:

```python
# Fit using Generalized Quadratic (implicit equation) - default
country_results_gq = fit_country_lorenz_curves(
    data_df,
    income_cols,
    n_params=3,
    curve_type='quadratic'
)

# Fit using Sarabia Ordered Family curve
country_results_sarabia = fit_country_lorenz_curves(
    data_df,
    income_cols,
    n_params=3,
    curve_type='sarabia'
)

# Fit using Beta Lorenz curve
country_results_beta = fit_country_lorenz_curves(
    data_df,
    income_cols,
    n_params=3,
    curve_type='beta'
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

#### `lorenz_gq_3`: Generalized Quadratic (default for `curve_type='quadratic'`)

Solves the implicit equation:
```
L(1-L) = a(p² - L) + bL(p-1) + c(p-L)
```

This reduces to a quadratic equation in L and selects the solution between 0 and 1.

**References:**
- Villasenor, J., and B. Arnold (1989). Elliptical Lorenz curves. Journal of Econometrics, 40, 327–338.
- https://rpubs.com/tsamuel/709170

#### `lorenz_sarabia_3`: Sarabia Ordered Family (`curve_type='sarabia'`)

```
L(p) = p^a · (1 - (1-p)^b)^c
```

With constraints: a > 0, b > 0, c > 1

**Reference:** Sarabia, J., E. Castillo and D. Slottje (1999). An Ordered Family of Lorenz Curves, Journal of Econometrics, 91, 43-60.

#### `lorenz_beta_3`: Beta Lorenz (`curve_type='beta'`)

```
L(p) = p - a · p^b · (1-p)^c
```

With constraints: 0 < a, b, c < 1

**Reference:** Kakwani, N. (1980). On a Class of Poverty Measures, Econometrica, 48, 437–446.

## Output

The workflow generates:

1. **country_results_Nparam.csv**: Fitted parameters and Gini coefficients for each country
2. **global_distribution.csv**: Global income distribution data
3. **country_lorenz_curves.png**: Sample country Lorenz curves
4. **global_lorenz_curve.png**: Fitted global Lorenz curve
5. **summary_report.txt**: Summary statistics and results

## Theory

### Lorenz Curves

A Lorenz curve L(p) represents the cumulative proportion of income held by the bottom p proportion of the population. Key properties:

- L(0) = 0: The bottom 0% has 0% of income
- L(1) = 1: The bottom 100% has 100% of income
- L is concave: Income inequality means the curve lies below the diagonal
- Gini coefficient = 1 - 2∫L(p)dp from 0 to 1

### Global Aggregation

The global distribution is computed by:

1. For each income threshold y
2. For each country with mean income μ_c
3. Calculate the fraction of country population with income < y using the country Lorenz curve
4. Sum across all countries weighted by population
5. Result: global cumulative distribution function

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
