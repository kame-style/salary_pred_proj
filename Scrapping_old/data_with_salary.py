# -*- coding: utf-8 -*-
"""
Created on Tue May  9 22:43:34 2023

@author: HP
"""

import pandas as pd 
import numpy as np

df=pd.read_csv('jobs.csv')

df1=df[df['salary'].isnull()]
     
print(df1.shape)