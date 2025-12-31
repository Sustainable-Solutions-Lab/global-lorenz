# Global Lorenz Curves

Fit global Lorenz curves to World Bank income distribution data using convex polynomial basis functions with optimized exponents.

## Features

### Primary Approach: Polynomial with Convex Combinations

- **Convex polynomial fitting**: Uses convex combinations of power basis functions (p^pᵢ) with optimized exponents
- **Automatic convexity guarantee**: Convex combination of convex functions ensures mathematical validity
- **Exceptional fit quality**: R² > 0.9999 with degree 4 polynomial
- **Closed-form Gini calculation**: Analytical formula for Gini coefficient from fitted parameters
- **Optimized exponent placement**: Nested optimization finds optimal power values, including very high powers (e.g., p^135) for tail accuracy
- **Flexible degree selection**: Fit polynomials from degree 1 to 6 or more
- **Simple interface**: Single command-line tool (`fit_polynomial_lorenz.py`) with minimal options

### Alternative Approach: Traditional Lorenz Curve Forms

- **Empirical global distribution**: Build high-resolution global income distribution by aggregating country-level data
- **Multiple parametric forms**: 5 traditional Lorenz curve types available:
  - 1-parameter: Pareto (`pareto_1`)
  - 2-parameter: Ortega/Jantzen-Volpert (`ortega_2`)
  - 3-parameter: Generalized Quadratic (`gq_3`), Beta (`beta_3`), Sarabia (`sarabia_3`)
- **Country-level fitting**: Fit parametric curves to individual countries for interpolation
- **Hybrid error objective**: Balances fitting quality across poor and rich populations
- **Comprehensive output**: CSV files with distribution data, visualization plots, and summary statistics

## Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Data Source

Primary data source: [World Bank Poverty and Inequality Platform (PIP) Poverty Calculator](https://pip.worldbank.org/poverty-calculator)

The data is stored in the Excel file in `./data/input/`.

## Data Preparation

Both the polynomial and traditional approaches use the same data preparation pipeline.

### Input Data Format

The input file must be in World Bank Poverty and Inequality Platform (PIP) format with these columns:

**Required columns:**
- **`decile1` ... `decile10`**: Income share for each decile (poorest 10% to richest 10%)
  - Values sum to 1.0 for each country-year observation
  - `decile1` = poorest 10%, `decile10` = richest 10%
- **`reporting_pop`**: Population for weighting
- **`reporting_gdp`**: GDP in PPP-consistent units
- **`country_name`**: Country name
- **`reporting_year`**: Year of observation

**Optional columns:**
- **`reporting_pce`**: Private consumption expenditure (PPP-consistent)

### Filtering to Most Recent Data

When the input contains multiple years per country, the pipeline automatically:
- Selects the most recent year with complete data (non-null values for all deciles, population, and GDP)
- Ensures each country has exactly one observation

This is performed by `filter_most_recent_complete()` from `global_lorenz.country_fitting`.

### Creating the Empirical Distribution

Before polynomial fitting, you must create the empirical global distribution:

```bash
python empirical_global_distribution.py
```

This script:
- Loads raw data from `data/input/pip_2025-12-28.xlsx`
- Filters to most recent complete data per country
- Treats each country-decile as a step function (everyone earns the mean for that decile)
- Aggregates across all countries to create global cumulative distribution
- Saves to `data/output/empirical_distribution_global.csv` (input for polynomial fitting)
- Calculates and reports empirical Gini coefficient

## Usage

### Primary Approach: Polynomial Fitting

#### Complete Workflow

**Step 1: Create empirical distribution** (see [Data Preparation](#data-preparation))
```bash
python empirical_global_distribution.py
```

**Step 2: Fit polynomial curves**
```bash
python fit_polynomial_lorenz.py --use-convex-combination --max-degree 6
```

This will:
- Load the empirical global Lorenz data from `data/output/empirical_distribution_global.csv`
- Fit polynomials with convex combinations for degrees 1 through 6
- Optimize both weights (wᵢ) and exponents (pᵢ) using nested optimization
- Generate output with fit statistics, parameters, and visualizations

**Output**: Best-fit polynomial with exceptional accuracy (R² > 0.9999 for degree 4)

#### Command-Line Options

```bash
python fit_polynomial_lorenz.py [options]
```

**Options:**

- `--use-convex-combination`: Use constrained convex combination (recommended)
  - Forces weights to sum to 1 and be non-negative
  - Guarantees convexity of resulting Lorenz curve
- `--max-degree=N`: Maximum polynomial degree to fit (default: 6)
  - Fits all degrees from 1 to N
  - Degree 4 typically provides excellent fit quality

**Example output (degree 4):**
```
L(p) = 0.1599·p^1.500 + 0.3776·p^4.367 + 0.3671·p^14.072 + 0.0954·p^135.060
R² = 0.999979
RMSE = 0.001026
Gini = 0.6812 (empirical: 0.6809)
```

### Alternative Approach: Traditional Lorenz Curve Forms

The traditional approach uses `main.py` to construct empirical global distributions and fit parametric Lorenz curves.

#### Quick Start

```bash
python main.py data/input/pip_2025-12-28.xlsx
```

This will:
- Read income distribution data from World Bank PIP format (Excel or CSV)
- Filter to the most recent year for each country with complete data
- Construct empirical global Lorenz curve by aggregating country distributions
- Fit all 5 traditional parametric curve types to the data
- Generate output in timestamped directories: `data/output/{lorenz_type}_{timestamp}/`

#### Basic Usage

```bash
python main.py <input_file> [lorenz_types] [options]
```

**Arguments:**

- `input_file` (required): Path to Excel/CSV file with income distribution data
- `lorenz_types` (optional): Comma-separated list of parametric curve types to fit
  - Options: `pareto_1`, `ortega_2`, `gq_3`, `beta_3`, `sarabia_3`
  - Default: all five types

**Key Options:**

- `--agg-bins=N`: Number of bins for high-resolution aggregation (default: 1000)
- `--fit-bins=N`: Number of equal-population bins for output (default: 100)
- `--error-type=TYPE`: Error metric (hybrid/absolute/fractional, default: hybrid)
- `--cumulative`: Fit on cumulative L(p) instead of income shares

**Examples:**
```bash
# Fit only beta_3 curve
python main.py data/input/pip_2025-12-28.xlsx beta_3

# Higher resolution empirical curve
python main.py data/input/pip_2025-12-28.xlsx --agg-bins=2000 --fit-bins=200

# Use absolute error metric
python main.py data/input/pip_2025-12-28.xlsx beta_3 --error-type=absolute
```

See the [Traditional Lorenz Curve Forms](#traditional-lorenz-curve-forms) section for details on the parametric forms.

For input data format and filtering details, see the [Data Preparation](#data-preparation) section.

### Programmatic Usage (Alternative Approach)

The traditional approach provides programmatic access for building empirical distributions:

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
income_cols = [f'decile{i}' for i in range(1, 11)]
data_df = filter_most_recent_complete(raw_data, income_cols)

# Fit country-level curves and build global distribution
country_results = fit_country_lorenz_curves(data_df, income_cols, 'ortega_2')
global_params, global_lorenz_func, global_gini, global_data = fit_global_lorenz(
    country_results, 'ortega_2', None
)

print(f"Global Gini coefficient: {global_gini:.4f}")
```

Choose different parametric forms: `pareto_1`, `ortega_2`, `gq_3`, `beta_3`, or `sarabia_3`.

## Polynomial Lorenz Curves with Convex Combinations

This is the recommended approach for fitting parametric Lorenz curves to empirical data. It uses convex combinations of power basis functions with optimized exponents.

### Method: Constrained Power Basis Functions

Uses a **convex combination of power basis functions** with optimized exponents. This method automatically guarantees convexity while achieving exceptional fit quality.

**Form:**
```
L(p) = w₀·p^p₀ + w₁·p^p₁ + w₂·p^p₂ + w₃·p^p₃
```

**Constraints:**
- Weights: wᵢ ≥ 0 and Σwᵢ = 1 (convex combination)
- Powers: pᵢ > 1 (ensures each basis function is convex)
- Boundary conditions: L(0) = 0 and L(1) = 1 (automatic)

**Convexity guarantee:** A non-negative weighted sum of convex functions is convex, so L(p) is guaranteed to be convex on [0,1].

### Best Fit (Degree 4)

The optimal fit using 4 basis functions with both weights and powers optimized:

```
L(p) = 0.1599·p^1.500 + 0.3776·p^4.367 + 0.3671·p^14.072 + 0.0954·p^135.060
```

**Fit Quality:**
- R² = 0.999979
- RMSE = 0.001026
- MAE = 0.000615
- Gini coefficient = 0.6812 (empirical: 0.6809)

**Analytical Gini Calculation:**

For this functional form, the Gini coefficient has a closed-form solution:

```
Gini = 1 - 2∫₀¹ L(s) ds = 1 - 2·Σᵢ [wᵢ/(pᵢ + 1)]
```

For the degree 4 fit:
```
Gini = 1 - 2·[0.1599/2.500 + 0.3776/5.367 + 0.3671/15.072 + 0.0954/136.060]
     = 1 - 2·[0.0639 + 0.0704 + 0.0244 + 0.0007]
     = 1 - 2·[0.1594]
     = 0.6812
```

### Fitting Tool

To fit polynomials with convex combinations:

```bash
python fit_polynomial_lorenz.py --use-convex-combination --max-degree 6
```

This uses nested optimization:
1. **Outer loop (nonlinear)**: Optimizes the power values pᵢ
2. **Inner loop (linear)**: For given powers, finds optimal weights wᵢ via constrained least squares

The method is superior to fixed linearly-spaced powers, finding optimal power placements including very high powers (e.g., p^135) that capture extreme inequality in the tail.

## Traditional Lorenz Curve Forms

This section describes the traditional parametric functional forms available as an alternative to the polynomial approach. These are used in `main.py` for country-level and global fitting.

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

### Polynomial Fitting Output

`fit_polynomial_lorenz.py` generates output to the console showing fit quality for each degree:

```
Degree 4 Polynomial Fit:
L(p) = 0.1599·p^1.500 + 0.3776·p^4.367 + 0.3671·p^14.072 + 0.0954·p^135.060

Fit Quality:
  R² = 0.999979
  RMSE = 0.001026
  MAE = 0.000615

Gini Coefficient:
  Fitted: 0.6812
  Empirical: 0.6809
  Error: 0.0003
```

The tool fits all degrees from 1 to `--max-degree` and displays results for comparison.

### Traditional Approach Output

`main.py` generates timestamped output directories (e.g., `data/output/ortega_2_20251228-140106/`) containing:

1. **`global_distribution_{lorenz_type}.csv`**: Empirical global Lorenz curve
   - `population_fraction`, `income_fraction`, `income_share`
   - Typically ~100 equal-population bins

2. **`global_lorenz_curve_{lorenz_type}.png`**: Visualization with linear and log-scale views

3. **`country_results_{lorenz_type}.csv`**: Country-level parametric fit results
   - Country, year, Gini coefficient
   - Fitted parameters and goodness-of-fit statistics (RMSE, MAE)
   - Actual vs fitted decile shares

4. **`summary_report_{lorenz_type}.txt`**: Summary statistics
   - Global Gini coefficient
   - Country-level fit quality statistics

## Theory

### Lorenz Curves

A Lorenz curve L(p) represents the cumulative proportion of income held by the bottom p proportion of the population. Key properties:

- L(0) = 0: The bottom 0% has 0% of income
- L(1) = 1: The bottom 100% has 100% of income
- L is concave: Income inequality means the curve lies below the diagonal
- Gini coefficient = 1 - 2∫L(p)dp from 0 to 1

### Polynomial Fitting with Convex Combinations

The recommended approach uses **convex combinations of power basis functions** with optimized exponents.

**Mathematical form:**
```
L(p) = Σᵢ wᵢ · p^pᵢ
```

**Constraints:**
- Weights: wᵢ ≥ 0 and Σwᵢ = 1 (convex combination)
- Powers: pᵢ > 1 (ensures each basis function is convex)
- Boundary conditions: L(0) = 0 and L(1) = 1 (automatic)

**Convexity guarantee:** A non-negative weighted sum of convex functions is convex, so L(p) is guaranteed to be convex on [0,1].

**Nested optimization:**
1. **Outer loop (nonlinear)**: Optimizes the power values pᵢ
2. **Inner loop (linear)**: For given powers, finds optimal weights wᵢ via constrained least squares

**Closed-form Gini coefficient:**
```
Gini = 1 - 2∫₀¹ L(s) ds = 1 - 2·Σᵢ [wᵢ/(pᵢ + 1)]
```

This approach achieves R² > 0.9999 with degree 4 polynomials while guaranteeing mathematical validity.

### Alternative: Empirical Distribution Construction

The traditional approach (`main.py`) constructs an empirical global Lorenz curve by aggregating country-level distributions:

1. Create 1000 log-spaced income thresholds
2. For each threshold and country, find population and income fractions using country Lorenz curves
3. Aggregate across countries with population weighting
4. Resample to 100 equal-population bins

Traditional parametric forms (Pareto, Ortega, Generalized Quadratic, Beta, Sarabia) can be fitted using a hybrid error objective that balances absolute and relative errors.

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
