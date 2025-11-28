"""
Evaluation metrics function for model predictions.

This module provides functionality to compute various performance metrics
for predictive models, particularly useful for insurance premium predictions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import auc


def evaluate_predictions(predictions, actuals, sample_weight):
    """
    Compute various evaluation metrics for model predictions.
    
    This function calculates multiple performance metrics to assess the quality
    of model predictions, particularly designed for insurance pricing models where
    exposure weighting is important.
    
    Parameters
    ----------
    predictions : array-like
        Model predictions (e.g., predicted pure premium)
    actuals : array-like
        True outcome values (e.g., actual pure premium)
    sample_weight : array-like
        Sample weights for each observation (e.g., exposure)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metric names as index and computed values
        
    Examples
    --------
    >>> predictions = np.array([100, 200, 150])
    >>> actuals = np.array([120, 180, 160])
    >>> weights = np.array([0.5, 1.0, 0.8])
    >>> metrics = evaluate_predictions(predictions, actuals, weights)
    """
    # Convert to numpy arrays for consistent handling
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    sample_weight = np.asarray(sample_weight)
    
    # Initialize metrics dictionary
    metrics = {}
    
    # 1. Bias: Deviation from actual exposure-adjusted mean
    # Bias = (sum(predictions * weight) - sum(actuals * weight)) / sum(weight)
    weighted_pred_sum = np.sum(predictions * sample_weight)
    weighted_actual_sum = np.sum(actuals * sample_weight)
    total_weight = np.sum(sample_weight)
    
    bias = (weighted_pred_sum - weighted_actual_sum) / total_weight
    metrics['Bias'] = bias
    
    # 2. Deviance: Tweedie deviance with power=1.5 (common for insurance)
    # Deviance = 2 * sum(weight * (y**(2-p)/(1-p)/(2-p) - y*y_pred**(1-p)/(1-p) + y_pred**(2-p)/(2-p)))
    # For Tweedie with p=1.5
    p = 1.5
    
    # Handle zeros carefully to avoid division by zero
    # Tweedie deviance formula for p=1.5:
    # d = 2 * [y^(2-p)/(1-p)/(2-p) - y*y_pred^(1-p)/(1-p) + y_pred^(2-p)/(2-p)]
    
    # For p=1.5: (2-p)=0.5, (1-p)=0.5
    # Simplified: d = 2 * [y^0.5/(-0.5*0.5) - y*y_pred^(-0.5)/(-0.5) + y_pred^0.5/0.5]
    # Which is: d = 2 * [-4*y^0.5 + 2*y/y_pred^0.5 + 2*y_pred^0.5]
    
    # Avoid division by zero
    predictions_safe = np.maximum(predictions, 1e-10)
    actuals_safe = np.maximum(actuals, 0)  # Actuals can be zero (no claims)
    
    deviance_components = (
        -4 * np.power(actuals_safe, 0.5)
        + 2 * actuals_safe / np.power(predictions_safe, 0.5)
        + 2 * np.power(predictions_safe, 0.5)
    )
    
    deviance = 2 * np.sum(sample_weight * deviance_components) / total_weight
    metrics['Deviance'] = deviance
    
    # 3. MAE: Mean Absolute Error (exposure-weighted)
    absolute_errors = np.abs(predictions - actuals)
    mae = np.sum(sample_weight * absolute_errors) / total_weight
    metrics['MAE'] = mae
    
    # 4. RMSE: Root Mean Squared Error (exposure-weighted)
    squared_errors = (predictions - actuals) ** 2
    mse = np.sum(sample_weight * squared_errors) / total_weight
    rmse = np.sqrt(mse)
    metrics['RMSE'] = rmse
    
    # 5. Gini coefficient (from Lorenz curve)
    # Based on the lorenz_curve function in ps3_script.py
    gini = compute_gini(actuals, predictions, sample_weight)
    metrics['Gini'] = gini
    
    # Convert to DataFrame with metrics as index
    metrics_df = pd.DataFrame(metrics, index=['Value']).T
    metrics_df.columns = ['Value']
    
    return metrics_df


def compute_gini(y_true, y_pred, exposure):
    """
    Compute Gini coefficient from Lorenz curve.
    
    The Gini coefficient measures the model's ability to rank predictions,
    where higher values indicate better ranking ability.
    
    Parameters
    ----------
    y_true : array-like
        True outcome values
    y_pred : array-like
        Predicted values
    exposure : array-like
        Exposure weights
        
    Returns
    -------
    float
        Gini coefficient (0 to 1, higher is better)
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)
    
    # Order samples by increasing predicted risk
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    
    # Compute cumulative claim amounts
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount = cumulated_claim_amount / cumulated_claim_amount[-1]
    
    # Create cumulative samples (x-axis for Lorenz curve)
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    
    # Gini coefficient: 1 - 2 * AUC(Lorenz curve)
    gini = 1 - 2 * auc(cumulated_samples, cumulated_claim_amount)
    
    return gini