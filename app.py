# -*- coding: utf-8 -*-
"""
Created on Mon May 29 23:50:31 2023

@author: HP
"""

from flask import Flask, request, jsonify, render_template
import pickle 
import numpy as np   
import pandas as pd 


app = Flask(__name__)


model  = pickle.load(open('model_file.pkl', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))
ohe    = pickle.load(open('ohe.pickle','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sample=[]
    for x in request.form.values():
        sample.append(x)
    print("initial")
    print(len(sample))
    print(type(sample))
    print(sample)
    
    sample=np.array(sample).reshape(1,-1)
    print("reshape")
    print(type(sample))
    print(sample)
    sample=pd.DataFrame(sample,columns=[['Rating', 'Size', 'Type of ownership', 'Sector', 'Revenue',
                                          'num_comp', 'hourly', 'employer_provided','job_state',
                                          'same_state','age', 'python_yn', 'spark', 'aws', 'excel',
                                          'job_simp', 'seniority','desc_len']])
    print("after dataframe")
    print(sample.shape)
    print(type(sample))
    print(sample)
    cat_attribs = ['Type of ownership', 'Sector', 'Revenue','job_state','job_simp','seniority']   ## 6

    num_attribs = ['Rating', 'Size', 'num_comp', 'hourly', 'employer_provided', 'same_state', 'age', 'python_yn',
                   'spark', 'aws', 'excel','desc_len']        ## 12

    print("scaling")
    sample_num=scaler.transform(sample[num_attribs])
    print("encoding")
    sample_cat=ohe.transform(sample[cat_attribs])
    print("stacking")
    sample_clean=np.hstack((sample_num,sample_cat))
    print("predicting")
    prediction = model.predict(sample_clean)    
    output=prediction
        

    return render_template('index.html', prediction='The Predicted salary with these job features will be : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
