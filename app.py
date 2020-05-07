# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:09:25 2020

@author: sefa
"""
import pandas as pd
from flask import Flask,jsonify
from sklearn.externals import joblib

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/<int:PageCount>/<int:LineCount>/<int:PicCount>')
def index(PageCount,LineCount,PicCount):
    
    try:     
        
        array=pd.DataFrame([[PageCount,LineCount,PicCount]])
        
        array = scaler.transform(array)    
        print(model.predict(array)[0])    
        print(type(model.predict(array)[0]))          
        return jsonify(result=str(model.predict(array)[0]))
    
    except:
        return jsonify(result=str("error"))
           
if __name__ == '__main__':
    app.run(port=5000,debug=True)
