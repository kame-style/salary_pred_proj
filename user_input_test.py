# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:56:12 2023

@author: HP
"""

from model_building import ohe
import pandas as pd

cat_attribs = ['Type of ownership', 'Sector', 'Revenue','job_state','job_simp','seniority']   ## 6
num_attribs = list(X_train.drop(cat_attribs,axis=1).columns)                                    ## 12

x_user=[3.7, 10000, 'Company - Public','Personal Consumer Services','$5 to $10 billion (USD)',
        '0', '0', '0', 'NY', '1',
        '12', '1', '1', '1', '1', 'data_scientist', 'na','3000']

df_user=pd.dataframe(x_user,columns=[['Rating', 'Size', 'Type of ownership', 'Sector', 'Revenue', 'num_comp', 'hourly', 'employer_provided',
                                      'job_state', 'same_state','age', 'python_yn', 'spark', 'aws', 'excel', 'job_simp', 'seniority','desc_len']])
x_user_clean=ohe.transform(df_user[cat_attribs])
#print(test_x_clean.shape)

x_user_clean=np.hstack((df_user[num_attribs],x_user_clean))
#print(x_user_clean.shape)

tpred_rf_user = gs.best_estimator_.predict(x_user_clean)
print(tpred_rf_user*1000)