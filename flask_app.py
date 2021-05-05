import os
import pickle

from flask import Flask
from flask import render_template
from flask import request, jsonify, redirect

import mysklearn.myutils as myutils
from mysklearn.myclassifiers import MyKNeighborsClassifier, MySimpleLinearRegressor, MyNaiveBayesClassifier, MyZeroRClassifier, MyRandomClassifier, MyDecisionTreeClassifier


app = Flask(__name__)
star_colors = ["red", "blue", "other"]
spectral_classes = ["O", "B", "A", "F", "G", "K", "M"]

@app.route("/")
def index():
    return render_template("index.html", colors=star_colors, scs=spectral_classes)


# Temp, L (luminosity), R (radius), A_M (magnitute), Color, Spectral_Class -> Type
@app.route("/api/predict")
def get_api_prediction():
    temp = request.args.get("temp", "")
    lum = request.args.get("lum", "")
    rad = request.args.get("rad", "")
    a_m = request.args.get("mag", "")
    color = request.args.get("color", "")
    s_c = request.args.get("spc", "")

    # load classifier
    prediction = predict_star(temp, lum, rad, a_m, color, s_c)
    if prediction is not None:
        return jsonify({"star_type": prediction}), 200
    else:
        return "prediction failed", 400


@app.route("/predict")
def get_prediction():
    temp = request.args.get("temp", "")
    lum = request.args.get("lum", "")
    rad = request.args.get("rad", "")
    a_m = request.args.get("mag", "")
    color = request.args.get("color", "")
    s_c = request.args.get("spc", "")

    # load classifier
    prediction = predict_star(temp, lum, rad, a_m, color, s_c)
    if prediction is not None:
        return render_template("index.html", colors=star_colors, scs=spectral_classes, result=prediction)
    else:
        return render_template("index.html", colors=star_colors, scs=spectral_classes, result=-1)


# Helper functions
def predict_star(temperature, l, r, a_m, color, s_c):
    infile = open("classifier.p")
    classifier = pickle.load(infile)    # classifier is ____ object
    infile.close()

    try:
        instance = [[temperature, l, r, a_m, color, s_c]]
        prediction = classifier.predict(instance)
        return prediction
    except:
        return None



if __name__ == "__main__":
    app.run()