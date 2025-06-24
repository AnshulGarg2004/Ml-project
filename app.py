import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import Custon_data, Predict_pipeline 
from src.utils import load_obj
application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else: 
        data = Custon_data(
            gender= request.form.get('gender'),
            race_ethnicity= request.form.get('ethnicity'),
            parental_level_of_education= request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            reading_score = float(request.form.get('reading_score')),
            writing_score= float(request.form.get('writing_score')),
            test_preparation_course= request.form.get('test_preparation_course')
        )

        pred_data = data.get_data_as_data_frame()
        print(pred_data)

        pred_pipeline = Predict_pipeline()
        results = pred_pipeline.predict(pred_data)
        return render_template('home.html', results = results[0])
    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')