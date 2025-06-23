from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and encoders
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Feature list (order matters)
FEATURES = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'ethnicity', 'jaundice', 'austim',
    'contry_of_res', 'used_app_before', 'result', 'relation'
]

# For select fields, get possible values from encoders
select_options = {}
for col in ['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'relation']:
    le = encoders[col]
    select_options[col] = list(le.classes_)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Collect form data
        form_data = {}
        for feat in FEATURES:
            form_data[feat] = request.form.get(feat)
        # Preprocess
        input_data = []
        for feat in FEATURES:
            val = form_data[feat]
            if feat in encoders:
                val = encoders[feat].transform([val])[0]
            elif feat in [
                'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score'
            ]:
                val = int(val)
            elif feat == 'age':
                val = int(float(val))
            elif feat == 'result':
                val = float(val)
            input_data.append(val)
        # Predict
        arr = np.array(input_data).reshape(1, -1)
        pred = model.predict(arr)[0]
        prediction = 'Autism Risk' if pred == 1 else 'No Autism Risk'
    return render_template('index.html', features=FEATURES, select_options=select_options, prediction=prediction)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True) 