"""
Tests for the evaluate_predictions function.
"""

import numpy as np
import pandas as pd
import pytest

from ps3.evaluation import evaluate_predictions


def test_evaluate_predictions_basic():
    """Test basic functionality of evaluate_predictions."""
    # Simple test case
    predictions = np.array([100.0, 200.0, 150.0])
    actuals = np.array([120.0, 180.0, 160.0])
    weights = np.array([0.5, 1.0, 0.8])
    
    result = evaluate_predictions(predictions, actuals, weights)
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that all expected metrics are present
    expected_metrics = ['Bias', 'Deviance', 'MAE', 'RMSE', 'Gini']
    assert all(metric in result.index for metric in expected_metrics)
    
    # Check that values are numeric
    assert result['Value'].dtype in [np.float64, float]


def test_evaluate_predictions_perfect_model():
    """Test with perfect predictions (predictions == actuals)."""
    predictions = np.array([100.0, 200.0, 150.0, 175.0])
    actuals = predictions.copy()
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    
    result = evaluate_predictions(predictions, actuals, weights)
    
    # Bias should be zero for perfect predictions
    assert abs(result.loc['Bias', 'Value']) < 1e-10
    
    # MAE should be zero
    assert abs(result.loc['MAE', 'Value']) < 1e-10
    
    # RMSE should be zero
    assert abs(result.loc['RMSE', 'Value']) < 1e-10


def test_evaluate_predictions_with_zeros():
    """Test handling of zero values in actuals."""
    predictions = np.array([100.0, 200.0, 150.0, 175.0])
    actuals = np.array([0.0, 180.0, 0.0, 175.0])  # Some zeros
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    
    result = evaluate_predictions(predictions, actuals, weights)
    
    # Should not raise any errors
    assert isinstance(result, pd.DataFrame)
    assert all(np.isfinite(result['Value']))


def test_evaluate_predictions_bias():
    """Test bias calculation specifically."""
    # Case where predictions are consistently higher
    predictions = np.array([110.0, 210.0, 160.0])
    actuals = np.array([100.0, 200.0, 150.0])
    weights = np.array([1.0, 1.0, 1.0])
    
    result = evaluate_predictions(predictions, actuals, weights)
    
    # Bias should be positive (predictions > actuals)
    assert result.loc['Bias', 'Value'] > 0
    
    # Expected bias: (110 + 210 + 160 - 100 - 200 - 150) / 3 = 30/3 = 10
    expected_bias = 10.0
    assert abs(result.loc['Bias', 'Value'] - expected_bias) < 1e-10


def test_evaluate_predictions_different_weights():
    """Test that weights are properly applied."""
    predictions = np.array([100.0, 200.0])
    actuals = np.array([100.0, 200.0])
    
    # First with equal weights
    weights1 = np.array([1.0, 1.0])
    result1 = evaluate_predictions(predictions, actuals, weights1)
    
    # Then with different weights
    weights2 = np.array([2.0, 1.0])
    result2 = evaluate_predictions(predictions, actuals, weights2)
    
    # Both should have zero bias (perfect predictions)
    assert abs(result1.loc['Bias', 'Value']) < 1e-10
    assert abs(result2.loc['Bias', 'Value']) < 1e-10


def test_gini_coefficient_range():
    """Test that Gini coefficient is in valid range."""
    # Random test case
    np.random.seed(42)
    predictions = np.random.uniform(50, 250, 100)
    actuals = predictions + np.random.normal(0, 20, 100)
    actuals = np.maximum(actuals, 0)  # Ensure non-negative
    weights = np.random.uniform(0.1, 1.0, 100)
    
    result = evaluate_predictions(predictions, actuals, weights)
    
    # Gini should be between 0 and 1
    gini = result.loc['Gini', 'Value']
    assert 0 <= gini <= 1, f"Gini coefficient {gini} is out of valid range [0, 1]"


def test_mae_rmse_relationship():
    """Test that RMSE >= MAE (mathematical property)."""
    predictions = np.array([100.0, 200.0, 150.0, 175.0])
    actuals = np.array([110.0, 190.0, 160.0, 170.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    
    result = evaluate_predictions(predictions, actuals, weights)
    
    mae = result.loc['MAE', 'Value']
    rmse = result.loc['RMSE', 'Value']
    
    # RMSE should be >= MAE (due to squaring)
    assert rmse >= mae


if __name__ == "__main__":
    # Run basic tests
    test_evaluate_predictions_basic()
    test_evaluate_predictions_perfect_model()
    test_evaluate_predictions_with_zeros()
    test_evaluate_predictions_bias()
    test_evaluate_predictions_different_weights()
    test_gini_coefficient_range()
    test_mae_rmse_relationship()
    
    print("All tests passed! ✅")