# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 19:27:29 2025

@author: gvino
"""

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all values from form
        features = [float(x) for x in request.form.values()]
        final_input = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_input)[0]
        result = "at risk of heart failure 💔" if prediction == 1 else "not at risk ❤️"

        return render_template('index.html', prediction_text=f"The patient is {result}.")
    except:
        return render_template('index.html', prediction_text="Invalid input. Please check and try again.")

if __name__ == '__main__':
    app.run(debug=True)
