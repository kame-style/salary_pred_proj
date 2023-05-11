# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:08:03 2023

@author: HP
"""

import glassdoor_scrapper as gs 
import pandas as pd 

path = "D:/Study/MLDL/chromedriver"

df = gs.get_jobs('data scientist',1000, False, path, 15)



## df.to_csv('glassdoor_jobs.csv', index = False)