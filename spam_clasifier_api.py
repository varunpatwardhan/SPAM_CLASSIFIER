#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from flask import Flask, jsonify,request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
clf = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features =tfidf.transform(np.array([data['news']]))
    prediction =clf.predict(features)
    spam_probability = clf.predict_proba(features)[0][1]
    output = str(prediction[0])
    return jsonify(IsSpam = output,spam_probability =spam_probability)
if __name__ == '__main__':
     app.run(host= '0.0.0.0',port=8899)


# In[ ]:




