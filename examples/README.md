# Examples

This directory contains example scripts and notebooks demonstrating the use of the global-lorenz package.

## Files

### generate_sample_data.py

Generates synthetic country-level income distribution data for testing and demonstration.

**Usage:**
```bash
python generate_sample_data.py
```

This will create two Excel files in the `data/` directory:
- `synthetic_country_data.xlsx`: 50 countries with randomly generated distributions
- `realistic_country_data.xlsx`: 30 countries with realistic names and distributions

### example_notebook.ipynb

Jupyter notebook demonstrating the complete workflow:
1. Loading data
2. Fitting country-level Lorenz curves
3. Visualizing results
4. Aggregating to global distribution
5. Fitting global Lorenz curves
6. Comparing different functional forms

**Usage:**
```bash
jupyter notebook example_notebook.ipynb
```

## Quick Start

1. Generate sample data:
```bash
cd examples
python generate_sample_data.py
```

2. Run the main workflow:
```bash
cd ..
python main.py data/realistic_country_data.xlsx
```

3. Check results in `output_1param/`, `output_2param/`, and `output_3param/` directories.

## Expected Outputs

After running the examples, you should see:
- CSV files with fitted parameters
- PNG images showing Lorenz curves
- Summary reports with Gini coefficients
- Global income distribution data
