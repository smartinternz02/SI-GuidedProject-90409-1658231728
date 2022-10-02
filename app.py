import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__,template_folder='template')
model = pickle.load(open("modelradomforest.pkl","rb"))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]

    features = [np.array(float_features)]
    prediction = model.predict(features)
    prediction = prediction*0.849*100
    return render_template("index.html", prediction_text = "Chance of getting admission is {}%".format(prediction))
    
if __name__ == '__main__':
    app.run(debug=True)