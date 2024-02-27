from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the Ridge regression model and StandardScaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting forest fire occurrence
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Extracting data from the form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Scaling the input data
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        # Predicting using the model
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result=result[0])
    else:
        return render_template('home.html')

# Route for the home page with FWI prediction result
@app.route('/home.html')
def home():
    # Example of result calculation (replace with your logic)
    result = " "
    return render_template('home.html', result=result)

if __name__ == "__main__":
    app.run()
