#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on jul 2024

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""

import pandas as pd

def read_datafile(filename):
    """
    Reads a file, either .csv or .xlsx, into a DataFrame.

    If the last 4 characters are 'xlsx', it reads it as an Excel file.
    If the last 3 characters are 'csv', it reads it as a CSV file.

    Parameters
    ----------
    filename : str
        The name of the file (.csv or .xlsx).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the information within the file.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    if str(filename[-4:])=='xlsx':
        df = pd.read_excel(filename)
    
    elif str(filename[-3:])=='csv':
        df = pd.read_csv(filename)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
 
    
    return df
