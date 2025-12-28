"""
Test suite for global-lorenz package.
"""

import numpy as np
import pandas as pd
import pytest
from global_lorenz import (
    lorenz_pareto_1,
    lorenz_ortega_2,
    lorenz_gq_3,
    fit_lorenz_curve,
    fit_lorenz_curve_decile,
    fit_country_lorenz_curves,
)
from global_lorenz.country_fitting import prepare_lorenz_data


class TestLorenzCurves:
    """Test Lorenz curve functional forms."""
    
    def test_lorenz_pareto_1_boundaries(self):
        """Test 1-parameter Lorenz curve at boundaries."""
        assert lorenz_pareto_1(0, 0.5) == 0.0
        assert lorenz_pareto_1(1, 0.5) == 1.0

    def test_lorenz_ortega_2_boundaries(self):
        """Test 2-parameter Lorenz curve at boundaries."""
        assert abs(lorenz_ortega_2(0, 1.0, 1.0)) < 1e-6
        assert abs(lorenz_ortega_2(1, 1.0, 1.0) - 1.0) < 1e-6

    def test_lorenz_gq_3_boundaries(self):
        """Test 3-parameter Lorenz curve at boundaries."""
        assert abs(lorenz_gq_3(0, 1.0, 1.0, 1.0)) < 1e-6
        assert abs(lorenz_gq_3(1, 1.0, 1.0, 1.0) - 1.0) < 1e-6

    def test_lorenz_monotonicity(self):
        """Test that Lorenz curves are monotonically increasing."""
        p = np.linspace(0, 1, 100)

        # 1-param
        L1 = lorenz_pareto_1(p, 0.5)
        assert np.all(np.diff(L1) >= 0)

        # 2-param
        L2 = lorenz_ortega_2(p, 1.5, 1.0)
        assert np.all(np.diff(L2) >= 0)

        # 3-param
        L3 = lorenz_gq_3(p, 1.0, 1.5, 1.0)
        assert np.all(np.diff(L3) >= 0)

    def test_lorenz_convexity(self):
        """Test that Lorenz curves are convex (below diagonal)."""
        p = np.linspace(0, 1, 100)

        # 1-param
        L1 = lorenz_pareto_1(p, -0.5)
        assert np.all(L1 <= p + 1e-6)  # Allow small numerical error

        # 2-param
        L2 = lorenz_ortega_2(p, 1.5, 1.0)
        assert np.all(L2 <= p + 1e-6)

        # 3-param
        L3 = lorenz_gq_3(p, 1.2, 1.5, 1.0)
        assert np.all(L3 <= p + 1e-6)


class TestFitting:
    """Test Lorenz curve fitting routines."""
    
    def test_prepare_lorenz_data(self):
        """Test conversion of income shares to Lorenz data."""
        shares = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.15, 0.15, 0.10, 0.07, 0.03])
        p, L = prepare_lorenz_data(shares)
        
        assert len(p) == len(shares) + 1
        assert len(L) == len(shares) + 1
        assert p[0] == 0 and p[-1] == 1
        assert L[0] == 0 and abs(L[-1] - 1.0) < 1e-6
        assert np.all(np.diff(L) >= 0)
    
    def test_fit_lorenz_pareto_1(self):
        """Test fitting 1-parameter Lorenz curve."""
        # Create synthetic data
        p_true = np.linspace(0, 1, 11)
        L_true = lorenz_pareto_1(p_true, -0.3)

        params, lorenz_func, gini, rmse = fit_lorenz_curve(p_true, L_true, 'pareto_1')

        assert len(params) == 1
        assert abs(params[0] - (-0.3)) < 0.1  # Should be close
        assert rmse < 0.01

    def test_fit_lorenz_ortega_2(self):
        """Test fitting 2-parameter Lorenz curve."""
        # Create synthetic data
        p_true = np.linspace(0, 1, 11)
        L_true = lorenz_ortega_2(p_true, 1.5, 1.2)

        params, lorenz_func, gini, rmse = fit_lorenz_curve(p_true, L_true, 'ortega_2')

        assert len(params) == 2
        assert rmse < 0.01

    def test_fit_lorenz_gq_3(self):
        """Test fitting 3-parameter Lorenz curve."""
        # Create synthetic data
        p_true = np.linspace(0, 1, 11)
        L_true = lorenz_gq_3(p_true, 1.0, 1.5, 1.0)

        params, lorenz_func, gini, rmse = fit_lorenz_curve(p_true, L_true, 'gq_3')

        assert len(params) == 3
        assert rmse < 0.01

    def test_gini_calculation(self):
        """Test Gini coefficient calculation."""
        # Perfect equality: L(p) = p, Gini = 0
        p = np.linspace(0, 1, 11)
        L = p.copy()
        params, _, gini, _ = fit_lorenz_curve(p, L, 'pareto_1')
        assert abs(gini) < 0.05  # Should be close to 0

    def test_fit_lorenz_decile_ortega_2(self):
        """Test fitting with fractional error objective."""
        # Create synthetic decile data
        income_shares = np.array([0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.25])

        params, lorenz_func, gini, rmse = fit_lorenz_curve_decile(income_shares, 'ortega_2')

        # Check that parameters were fitted
        assert len(params) == 2

        # Verify the fit is reasonable by checking predicted shares match actual shares
        total_error = 0.0
        for i in range(10):
            p_lower = i / 10
            p_upper = (i + 1) / 10
            L_lower = lorenz_func(p_lower, *params)
            L_upper = lorenz_func(p_upper, *params)
            predicted_share = L_upper - L_lower
            actual_share = income_shares[i]
            fractional_error = abs((predicted_share / actual_share) - 1.0)
            total_error += fractional_error

        # Average fractional error should be small
        avg_fractional_error = total_error / 10
        assert avg_fractional_error < 0.1  # Less than 10% average error


class TestCountryFitting:
    """Test country-level fitting routines."""
    
    def test_fit_country_lorenz_curves(self):
        """Test fitting Lorenz curves to multiple countries."""
        # Create synthetic country data with realistic income distributions
        data = []
        np.random.seed(42)
        for i in range(5):
            # Generate income shares that represent inequality (poorest has least)
            # Use power law to ensure proper ordering
            alpha = 1.0 + np.random.uniform(0.5, 1.5)
            shares = np.array([(j+1)**alpha for j in range(10)])
            shares = shares / shares.sum()  # Normalize
            
            row = {'Country': f'Country_{i}', 'GDP': 1e12, 'Population': 1e8}
            for j, share in enumerate(shares):
                row[f'D{j+1}'] = share
            data.append(row)
        
        df = pd.DataFrame(data)
        income_cols = [f'D{i}' for i in range(1, 11)]

        results = fit_country_lorenz_curves(df, income_cols, 'ortega_2')

        assert len(results) == 5
        assert 'country' in results.columns
        assert 'gini' in results.columns
        assert 'param_1' in results.columns
        assert 'param_2' in results.columns
        # Allow some tolerance for numerical issues
        assert all(results['gini'] >= -0.05) and all(results['gini'] <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
