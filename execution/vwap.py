# VWAP -- Volume Weighted Average Price

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def vwap(prices, volumes):
    """
    Calculate the Volume Weighted Average Price (VWAP).
    
    Parameters:
    prices (array-like): List or array of prices.
    volumes (array-like): List or array of volumes corresponding to the prices.
    
    Returns:
    float: The VWAP value.
    """
    prices = np.array(prices)
    volumes = np.array(volumes)
    
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have the same length.")
    
    vwap_value = np.sum(prices * volumes) / np.sum(volumes)
    return vwap_value