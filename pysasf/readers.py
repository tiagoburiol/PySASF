#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on jul 2024

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""

import pandas as pd

def read_datafile(filename):
    if str(filename[-4:])=='xlsx':
        df = pd.read_excel(filename)
        
    if str(filename[-3:])=='csv':
        df = pd.read_csv(filename)
    
    return df
