# Global Lorenz Curves

Construct empirical global Lorenz curves from World Bank income distribution data, with optional parametric curve fitting.

## Features

### Core Functionality

- **Empirical global Lorenz curve construction**: Build high-resolution global income distribution by aggregating country-level data
- **High-resolution aggregation**: Track global distribution at 1000 income thresholds, capturing detailed inequality patterns
- **Population-weighted aggregation**: Correctly combine country distributions using both population and income data
- **Equal-population resampling**: Convert to 100 percentile bins for stable analysis and visualization
- **Comprehensive output**: CSV files with empirical global distribution data, visualization plots, and summary statistics
- **Dual-scale visualization**: Linear and log-scale plots revealing inequality across entire income range
- **Poverty metrics**: Calculate global poverty headcount at various poverty lines

### Optional Parametric Fitting

- **Country-level fitting**: Fit parametric Lorenz curves to individual countries to smooth data and enable interpolation
- **Multiple functional forms**: 5 different Lorenz curve types available:
  - 1-parameter: Pareto (`pareto_1`)
  - 2-parameter: Ortega/Jantzen-Volpert (`ortega_2`)
  - 3-parameter: Generalized Quadratic (`gq_3`), Beta (`beta_3`), Sarabia (`sarabia_3`)
- **Global parametric fitting**: Fit smooth parametric curves to empirical global distribution
- **Hybrid error objective**: Automatically balances fitting quality across poor and rich populations
- **Goodness-of-fit statistics**: RMSE, mean absolute error, and maximum absolute error metrics
- **Flexible selection**: Run all curve types or select specific ones via command-line arguments

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

## Usage

### Quick Start

```bash
python main.py data/input/pip_2025-12-28.xlsx
```

This will automatically:
- Read income distribution data from World Bank PIP format (Excel or CSV)
- Filter to the most recent year for each country with complete data
- Construct empirical global Lorenz curve by aggregating country distributions
- Optionally fit parametric curves (default: all 5 types) to smooth the data
- Generate output in timestamped directories: `data/output/{lorenz_type}_{error_type}_{fit_method}_{timestamp}/`

**Primary output**: High-resolution empirical global Lorenz curve with ~100 percentile bins

**Secondary output**: Parametric curve fits for comparison and interpolation

### Command-Line Options

#### Basic Usage

```bash
python main.py <input_file> [lorenz_types] [options]
```

**Arguments:**

- `input_file` (required): Path to Excel/CSV file with income distribution data
- `lorenz_types` (optional): Comma-separated list of parametric curve types to fit
  - Options: `pareto_1`, `ortega_2`, `gq_3`, `beta_3`, `sarabia_3`
  - Default: all five types
  - **Note**: The empirical curve is always generated regardless of this setting

**Options:**

- `--agg-bins=N` (optional): Number of bins for high-resolution empirical aggregation
  - Default: `1000`
  - Controls resolution of empirical global distribution
- `--fit-bins=N` (optional): Number of equal-population bins for empirical curve output
  - Default: `100`
  - Controls number of percentile bins in final empirical curve
- `--error-type=TYPE` (optional): Error metric for parametric curve fitting
  - Options: `hybrid`, `absolute`, `fractional`
  - Default: `hybrid`
  - Only affects parametric fitting, not empirical curve
- `--cumulative` (optional): Fit parametric curves on cumulative L(p) instead of income shares
  - Default: fit on income shares
  - Only affects parametric fitting, not empirical curve

#### Examples

**Basic empirical curve construction:**
```bash
# Generate empirical global Lorenz curve with default settings
python main.py data/input/pip_2025-12-28.xlsx

# Higher resolution empirical curve (2000 aggregation bins, 200 output bins)
python main.py data/input/pip_2025-12-28.xlsx --agg-bins=2000 --fit-bins=200
```

**Parametric fitting options:**
```bash
# Fit only beta_3 parametric curve to empirical data
python main.py data/input/pip_2025-12-28.xlsx beta_3

# Fit multiple specific parametric forms
python main.py data/input/pip_2025-12-28.xlsx ortega_2,beta_3

# Use different error metric for parametric fitting
python main.py data/input/pip_2025-12-28.xlsx beta_3 --error-type=absolute

# Fit parametric curves on cumulative L(p) values
python main.py data/input/pip_2025-12-28.xlsx beta_3 --cumulative
```

**Combined options:**
```bash
# High-resolution empirical curve with single parametric fit
python main.py data/input/pip_2025-12-28.xlsx beta_3 --agg-bins=2000 --fit-bins=200

# All options combined
python main.py data/input/pip_2025-12-28.xlsx beta_3 --error-type=absolute --cumulative --agg-bins=500 --fit-bins=50
```

#### Parametric Fitting Strategies

When fitting parametric curves to the empirical distribution, the package supports two orthogonal dimensions:

1. **Fitting Target** (what to fit):
   - **Income shares** (default): Minimizes errors on income in each bin (approximately fitting dL/dp)
   - **Cumulative L(p)** (`--cumulative`): Minimizes errors on the Lorenz curve itself, preventing error accumulation across bins

2. **Error Metric** (how to measure error):
   - **Hybrid** (default): Minimizes product of absolute and relative errors = error²/value
   - **Absolute**: Minimizes absolute errors
   - **Fractional**: Minimizes relative/fractional errors

These options are **orthogonal** - you can combine any fitting target with any error metric:

- **Default (shares + hybrid)**: Balanced fit across all income levels, good general-purpose choice
- **Cumulative + hybrid**: Prevents error accumulation in L(p), good for applications using Lorenz curve values directly
- **Shares + fractional**: Emphasizes relative accuracy in each bin
- **Cumulative + absolute**: Minimizes absolute deviation of Lorenz curve from empirical data

**Note:** These options only affect parametric curve fitting. The empirical curve construction is always the same.

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

#### Basic Empirical Curve Construction

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

# Fit country-level Lorenz curves to enable interpolation
# This uses parametric curves to smooth country data
country_results = fit_country_lorenz_curves(
    data_df,
    income_cols,
    'ortega_2'  # 2-parameter form, good general-purpose choice
)

# Build empirical global Lorenz curve
# This aggregates country distributions at high resolution
global_params, global_lorenz_func, global_gini, global_data = fit_global_lorenz(
    country_results,
    'ortega_2',
    None
)

# The empirical global distribution is in global_data
# global_gini is calculated from the empirical distribution
print(f"Global Gini coefficient (empirical): {global_gini:.4f}")
```

#### Optional: Parametric Fitting

You can choose different parametric forms for country-level smoothing:

```python
# 3-parameter forms provide more flexibility

# Generalized Quadratic (implicit equation)
country_results_gq = fit_country_lorenz_curves(data_df, income_cols, 'gq_3')

# Sarabia Ordered Family
country_results_sarabia = fit_country_lorenz_curves(data_df, income_cols, 'sarabia_3')

# Beta Lorenz curve
country_results_beta = fit_country_lorenz_curves(data_df, income_cols, 'beta_3')

# 1-parameter Pareto (least flexible)
country_results_pareto = fit_country_lorenz_curves(data_df, income_cols, 'pareto_1')
```

## Alternative Parametric Form: Polynomial with Convex Combinations

This section describes an alternative parametric approach using polynomial basis functions instead of the traditional Lorenz curve forms.

### Constrained Power Basis Functions

This approach uses a **convex combination of power basis functions** with optimized exponents. This method automatically guarantees convexity while achieving exceptional fit quality to the empirical distribution.

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

## Parametric Lorenz Curve Forms

This section describes the parametric functional forms available for optional country-level and global fitting. These provide smooth approximations to the empirical distributions.

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

### Primary Output: Empirical Global Distribution

1. **`global_distribution_{lorenz_type}.csv`**: High-resolution empirical global Lorenz curve
   - `population_fraction`: Cumulative population fraction (p)
   - `income_fraction`: Cumulative income fraction L(p)
   - `income_share`: Income share for each percentile bin
   - Typically ~100 equal-population bins
   - This is the core output - the empirical global income distribution

2. **`global_lorenz_curve_{lorenz_type}.png`**: Visualization of empirical global Lorenz curve
   - Shows both linear and log-scale views
   - Reveals inequality patterns across entire income range
   - Displays empirical Gini coefficient

### Secondary Output: Parametric Fits

3. **`country_results_{lorenz_type}.csv`**: Country-level parametric fit results with columns:
   - `country`: Country name
   - `year`: Reporting year for the data
   - `gini`: Gini coefficient
   - `param_1`, `param_2`, `param_3`: Fitted Lorenz curve parameters
   - `gdp`, `population`: Country economic data
   - `decile1_actual` ... `decile10_actual`: Input income shares
   - `decile1_fitted` ... `decile10_fitted`: Predicted income shares from parametric curve
   - `rmse`, `mafe`, `max_abs_error`: Goodness-of-fit statistics

4. **`country_lorenz_curves_{lorenz_type}.png`**: Sample country parametric fit visualizations

5. **`summary_report_{lorenz_type}.txt`**: Summary statistics including:
   - Global Gini coefficient (from empirical distribution)
   - Country-level Gini statistics (mean, median)
   - Country-level parametric fit quality (RMSE statistics)
   - Parametric curve parameter values

### Interpreting Parametric Fit Statistics

All error metrics are **fractional (relative) errors**, not absolute errors:

- **RMSE = 0.05**: ~5% root mean squared fractional error across deciles
- **MAFE = 0.03**: On average, parametric predictions differ from actual by 3%
- **max_abs_error = 0.15**: The worst-fit decile has 15% relative error

**Example:** If a decile has actual share = 0.05 and fitted share = 0.055:
- Fractional error = (0.055/0.05) - 1 = 0.10 (10% error)

These statistics measure how well parametric curves approximate the country-level data. The empirical global distribution does not involve fitting and has no approximation error.

## Theory

### Lorenz Curves

A Lorenz curve L(p) represents the cumulative proportion of income held by the bottom p proportion of the population. Key properties:

- L(0) = 0: The bottom 0% has 0% of income
- L(1) = 1: The bottom 100% has 100% of income
- L is concave: Income inequality means the curve lies below the diagonal
- Gini coefficient = 1 - 2∫L(p)dp from 0 to 1

### Empirical Global Distribution Construction

The core methodology constructs an empirical global Lorenz curve by aggregating country-level distributions. This produces a high-resolution global income distribution without parametric assumptions.

**Algorithm:**

1. **High-resolution sampling**: Create 1000 income thresholds spanning the global income range (log-spaced)

2. **Country-level interpolation**: For each income threshold y and each country c:
   - Convert y to fraction of country mean income: y/μ_c
   - Find population fraction below y using country's Lorenz curve (parametric or empirical)
   - Find income fraction below y using country's Lorenz curve: L(p)
   - Add to global totals: cumulative_pop += p_c × pop_c, cumulative_income += L(p_c) × income_c

3. **Global Lorenz curve**: Normalize cumulative values to get global (p, L) with ~1000 data points

4. **Equal-population resampling**: Convert to 100 equal-population bins (percentiles) for stable output

5. **Gini calculation**: Compute global Gini coefficient directly from empirical distribution

**Key features:**
- **No parametric assumptions**: The global distribution is empirical, constructed by aggregation
- **Population-weighted**: Correctly accounts for both population and income across countries
- **High resolution**: 1000 thresholds capture detailed inequality patterns
- **Numerically stable**: Equal-population bins avoid issues with tiny bins in tails

### Optional: Parametric Fitting Methodology

Parametric curves can be fitted at two levels:

**Country-level fitting** (enables interpolation within countries):
- Uses 10 equal-population bins (deciles) from input data
- Fits parametric Lorenz curves to smooth and interpolate country distributions
- Uses hybrid error objective that balances fitting quality across income levels

**Hybrid error objective:**
```
Minimize: Σᵢ population_shareᵢ · (predicted_shareᵢ - actual_shareᵢ)² / actual_shareᵢ
```

This minimizes the **product of absolute and relative errors**:
- **Balanced importance**: Automatically balances fitting quality across poor and rich populations
- **Natural weighting**: Division by `actual_shareᵢ` emphasizes relative accuracy for small bins
- **Numerical stability**: Avoids division-by-zero while emphasizing fit quality everywhere

**Mathematical equivalence:**
```
Hybrid error² = (absolute_error)² / actual_share
              = (relative_error)² × actual_share
```

**Global parametric fitting** (optional smoothing):
- Can fit parametric curves to the empirical global distribution
- Uses the same hybrid error objective
- Provides smooth functional form for interpolation and extrapolation
- Secondary to the empirical distribution

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
