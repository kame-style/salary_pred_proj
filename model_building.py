# -*- coding: utf-8 -*-
"""
Created on Sun May 14 23:59:26 2023

@author: HP
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import re


df = pd.read_csv('eda_data.csv')

# choose relevant columns 
df.columns

df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]

# Some more DATA CLEANING

df_model= df_model.drop('Industry', axis =1)
df_model['Size'] = df_model['Size'].apply(lambda x: x.replace('employees',''))
df_model['Size'] = df_model['Size'].apply(lambda x: [int(s) for s in re.findall('\d+', x)])
df_model['Size'] = df_model['Size'].apply(lambda x: max(x,default=0))



# get dummy data 
# df_dum = pd.get_dummies(df_model)

# train test split 
from sklearn.model_selection import train_test_split

X = df_model.drop('avg_salary', axis =1)
y = df_model.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_attribs = ['Type of ownership', 'Sector', 'Revenue','job_state','job_simp','seniority']   ## 6

num_attribs = ['Rating', 'Size', 'num_comp', 'hourly', 'employer_provided', 'same_state', 'age', 'python_yn',
               'spark', 'aws', 'excel','desc_len']        ## 12

ohe= OneHotEncoder(sparse_output=False,dtype=np.int32,handle_unknown='ignore')
scaler= MinMaxScaler()
   
train_x_cat=ohe.fit_transform(X_train[cat_attribs])
print(train_x_cat.shape)
train_x_num=scaler.fit_transform(X_train[num_attribs])
print(train_x_num.shape)
train_x_clean=np.hstack((train_x_num,train_x_cat))
print(train_x_clean.shape)

test_x_cat=ohe.transform(X_test[cat_attribs])
print(test_x_cat.shape)
test_x_num=scaler.transform(X_test[num_attribs])
print(test_x_num.shape)
test_x_clean=np.hstack((test_x_num,test_x_cat))
print(test_x_clean.shape)

#%%
###############################################################################

'''
#Categoricals:
cat_attribs = ['Type of ownership', 'Industry', 'Revenue','job_state','job_simp','seniority']   ## 6

num_attribs = list(X_train.drop(cat_attribs,axis=1).columns)                                    ## 12


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion


class DataFrameSelector(BaseEstimator, TransformerMixin):
	""" this class will select a subset of columns,
		pass in the numerical or categorical columns as 
		attribute names to get just those columns for processing"""
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names]
    
class MultiColBinarize(BaseEstimator, TransformerMixin):
	""" take a df with multiple categoricals
		one hot encode them all and return the numpy array"""
	def __init__(self, alter_df= True):
		self.alter_df = alter_df
	def fit(self, X, y=None):
		"""load the data in, initiate the binarizer for each column"""
		self.X = X
		self.cols_list = list(self.X.columns)
		self.binarizers = []
		for i in self.cols_list:
			encoder = LabelBinarizer()
			encoder.fit(self.X[i])
			self.binarizers.append(encoder)
		return self
	def transform(self, X):
		""" for each of the columns, use the existing binarizer to make new cols """
		self.X = X
		self.binarized_cols = self.binarizers[0].transform(self.X[self.cols_list[0]])
		self.classes_ = list(self.binarizers[0].classes_)
		for i in range(1,len(self.cols_list)):
			binarized_col = self.binarizers[i].transform(self.X[self.cols_list[i]])
			self.binarized_cols = np.concatenate((self.binarized_cols , binarized_col), axis = 1)
			self.classes_.extend(list(self.binarizers[i].classes_))
		return self.binarized_cols

num_pipeline = Pipeline([
		('selector', DataFrameSelector(num_attribs)),
		('imputer', SimpleImputer(strategy="median")),
		('std_scaler', StandardScaler()),
	])
# select the categorical columns, binarize them 
cat_pipeline = Pipeline([
		('selector', DataFrameSelector(cat_attribs)),
		('label_binarizer', MultiColBinarize()),
	])

train_num_processed = num_pipeline.fit_transform(X_train)
train_cat_processed = cat_pipeline.fit_transform(X_train)

train_x_clean =  np.concatenate((train_num_processed,train_cat_processed),axis=1)

test_num_processed = num_pipeline.transform(X_test)
test_cat_processed = cat_pipeline.transform(X_test)

test_x_clean =  np.concatenate((test_num_processed,test_cat_processed),axis=1)

###############################################################################
'''
'''
# multiple linear regression 
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

#np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

# lasso regression 
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
#np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]
'''

###############################################################################
#%%
# random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=40)


from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(rf,train_x_clean,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

# tune models GridsearchCV 
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10),  'max_features':('sqrt','log2',1.0)}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(train_x_clean,y_train)

gs.best_score_
gs.best_estimator_

# test ensembles 
#tpred_lm = lm.predict(X_test)
#tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(test_x_clean)

from sklearn.metrics import mean_absolute_error
#mean_absolute_error(y_test,tpred_lm)
#mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf)

#mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)

######################################################################################################################################
##SAVING AND LOADING MODEL##

import pickle

pickle.dump(scaler,open("scaler.pickle","wb"))
pickle.dump(ohe,open("ohe.pickle","wb"))

#pickl = {'model': gs.best_estimator_}
#pickle.dump( pickl, open( "model_file.pkl", "wb" ) )

pickle.dump(gs.best_estimator_, open( "model_file.pkl", "wb" ) )

model=pickle.load(open("model_file.pkl", "rb" ))
sample_scaler=pickle.load(open("scaler.pickle","rb"))
sample_ohe=pickle.load(open("ohe.pickle","rb"))

sample=np.array(list(X_test.iloc[0,:])).reshape(1,-1)
print(list(X_test.iloc[0,:]))
#print(sample.dtypes)
#print(type(sample))
#print(sample.shape)
sample=pd.DataFrame(sample,columns=[['Rating', 'Size', 'Type of ownership', 'Sector', 'Revenue',
                                                                                'num_comp', 'hourly', 'employer_provided','job_state',
                                                                                'same_state','age', 'python_yn', 'spark', 'aws', 'excel',
                                                                                'job_simp', 'seniority','desc_len']])
#print(type(sample))
#print(sample.shape)
sample_num=scaler.transform(sample[num_attribs])
sample_cat=ohe.transform(sample[cat_attribs])
sample_clean=np.hstack((sample_num,sample_cat))
print(type(sample_clean))
print(sample_clean.shape)


#print(type(model))
ans=model.predict(sample_clean)
print("predicted : ", ans[0]*1000)
print("actual : ",y_test[0]*1000)

'''
file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

list(X_test.iloc[1,:])
'''
######################################################################################################################################
## USER INPUT ##

#X_test.columns
#['Rating', 'Size', 'Type of ownership', 'Sector', 'Revenue',
#       'num_comp', 'hourly', 'employer_provided', 'job_state', 'same_state',
#       'age', 'python_yn', 'spark', 'aws', 'excel', 'job_simp', 'seniority',
#       'desc_len']

df_user=X_test[0:0]
#df_user.shape
x_user=[3.7, 10000, 'Company - Public','Personal Consumer Services','$5 to $10 billion (USD)',
        '0', '0', '0', 'NY', '1',
        '12', '1', '1', '1', '1', 'data_scientist', 'na','3000']

df_user.loc[len(df.index)] = x_user
x_user_clean=ohe.transform(df_user[cat_attribs])
#print(test_x_clean.shape)

x_user_clean=np.hstack((df_user[num_attribs],x_user_clean))
#print(x_user_clean.shape)

tpred_rf_user = gs.best_estimator_.predict(x_user_clean)
print(tpred_rf_user*1000)

######################################################################################################################################



##Commands To productionize using FLASK
##
##change directory to current 
##>mkdir FlaskAPI  #to create a directory
##>cd FlaskAPI
##>conda create -n flask_env python=3.7
##>conda activate flask_env
##>conda install pandas
##>conda install scikit-learn
##>pip freeze > requirments.txt 
###create some fiels 
##>type nul >> "app.py"
##>type nul >> "Procfile"
##>type nul >> "wsgi.py"
##>mkdir models
###copy the code in appy.py and wsgi.py
##>pyhon wsgi.py #to start the server
##>from another anaconda prompt "curl -X GET http://0.0.0.0:8080/predict"
## move model file in FlaskAPI/models
##

