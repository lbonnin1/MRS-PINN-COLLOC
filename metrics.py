import torch
import numpy as np

def r_squared(y_true, y_pred):
    mean_observed = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_observed)**2)
    ss_res = np.sum((y_true - y_pred)**2)
   
    if ss_tot == 0:
        if np.allclose(y_true, y_pred):
            return 1.0  # Perfect fit case
        else:
            return float('nan')  # Undefined R-squared
    
    r2 = 1 - (ss_res / ss_tot)
    return r2
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE)
    
    Args:
        y_true (array-like): Array of actual values.
        y_pred (array-like): Array of predicted values.
        
    Returns:
        float: MAPE value.
    """
    nonzero_mask = y_true != 0
    y_true, y_pred = y_true[nonzero_mask], y_pred[nonzero_mask]
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def gaussian_area(amplitude, lwhm):
    # Constante sqrt(pi) / sqrt(ln(2))
    sqrt_pi = np.sqrt(np.pi)
    sqrt_ln2 = np.sqrt(np.log(2))
    
    # Calcul de l'aire sous la courbe gaussienne
    area = amplitude * (lwhm * sqrt_pi) / (2 * sqrt_ln2)
    
    return area

def lorentzian_area(amplitude):
    # Pour une lorentzienne, l'aire sous la courbe est simplement l'amplitude
    return amplitude


def draw_curve(x,amplitude,center,lwhm,type):
  
    if type == 0:
       return amplitude / (1 + ((x- center) / (lwhm/2) ) ** 2)
    else:
        return amplitude * np.exp(-4*np.log2(2) * ((x- center) / (lwhm))**2)

def mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    variance = np.var(y_true)
    return mse / variance