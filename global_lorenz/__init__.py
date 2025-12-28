"""
global-lorenz: Fit global Lorenz curves to World Bank Data

This package provides tools for:
1. Fitting Lorenz curves at the country level
2. Aggregating country-level data to global income distribution
3. Fitting global Lorenz curves using multiple functional forms
"""

__version__ = "0.1.0"

from .lorenz_curves import (
    lorenz_pareto_1,
    lorenz_ortega_2,
    lorenz_gq_3,
    lorenz_beta_3,
    lorenz_sarabia_3,
    fit_lorenz_curve,
)
from .country_fitting import fit_country_lorenz_curves
from .global_aggregation import aggregate_global_distribution, fit_global_lorenz

__all__ = [
    "lorenz_pareto_1",
    "lorenz_ortega_2",
    "lorenz_gq_3",
    "lorenz_beta_3",
    "lorenz_sarabia_3",
    "fit_lorenz_curve",
    "fit_country_lorenz_curves",
    "aggregate_global_distribution",
    "fit_global_lorenz",
]
