import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response
import pandas as pd
import datetime
from datetime import datetime


app = Flask(__name__, template_folder='template')

model = pickle.load(open('model.pkl', 'rb'))



@app.route("/")
def home():
    return render_template('Trail.html')

@app.route("/predict", methods = ['POST'])
def predict():
    values = [x for x in request.form.values()]
    final_features = np.array(values, dtype = 'str')
    prediction = model.predict(start = pd.to_datetime(final_features[0]), end = pd.to_datetime(final_features[1]), dynamics = False)
    return render_template('Trail.html', prediction_text = prediction)



if __name__ == "__main__":
    app.run(debug=False)
