import os
import pickle

from flask import Flask
from flask import render_template
from flask import request, jsonify, redirect

import mysklearn.myutils as myutils
from mysklearn.myclassifiers import MyDecisionTreeClassifier


app = Flask(__name__)
star_colors = ["red", "blue"]
spectral_classes = ["O", "B", "A", "F", "G", "K", "M"]

@app.route("/")
def index():
    return render_template("index.html", colors=star_colors, scs=spectral_classes)


@app.route("/test/<testp>")
def test_param(testp):
    return jsonify({"result": testp})

# Temp, L (luminosity), R (radius), A_M (magnitute), Color, Spectral_Class -> Type
@app.route("/api/predict", methods=['GET'])
def get_api_prediction():
    temp = request.args.get("temp", "")
    lum = request.args.get("lum", "")
    rad = request.args.get("rad", "")
    mag = request.args.get("mag", "")
    color = request.args.get("color", "")
    spc = request.args.get("spc", "")
    print("got all params")


    # load classifier
    prediction = predict_star(temp_bins(temp), luminosity_bins(lum), get_radius(rad), get_magnitude(mag), color, spc)
    if prediction is not None:
        return jsonify({"star_type": prediction}), 200
    else:
        return "prediction failed", 400



# Helper functions
def predict_star(temperature, l, r, a_m, color, s_c):
    infile = open("decision_tree.p", "rb")
    classifier = pickle.load(infile)    # classifier is ____ object
    infile.close()
    print("loaded classifier")

    try:
        instance = [[temperature, l, r, a_m, color, s_c]]
        prediction = classifier.predict(instance)
        return prediction
    except:
        return None


def temp_bins(data):
    data = int(data)
    if data < 5000:
        return "low"
    elif data < 10000:
        return "medium-low"
    elif data < 15000:
        return "medium"
    elif data < 20000:
        return "medium-high"
    else:
        return "high"
    


def luminosity_bins(data):
    data = int(data)
    if data < 85000:
        return "0-85000"
    elif data < 170000:
        return "85001-170000"
    elif data < 255000:
        return "170001-255000"
    elif data < 340000:
        return "255001-340000"
    else:
        return "greater than 340001"


def get_radius(data):
    data = int(data)
    if data < 100:
        return "0 - 100"
    elif data < 150:
        return "100.01 - 150"
    elif data < 200:
        return "150.01 - 200"
    else:
        return "> 200"


def get_magnitude(data):
    data = int(data)
    if data < -5:
        return "-11 - -5"
    elif data < 0:
        return "-4.99 - 0"
    elif data < 5:
        return "0.01 - 5"
    elif data < 10:
        return "5.01 - 10"
    elif data < 15:
        return "10.01 - 15"
    else:
        return "greater than 15"



def get_spectral_class(data):
    if data == 'M':
        return "M"
    elif data == 'B':
        return "B"
    else:
        return "other"
            



if __name__ == "__main__":
    app.run()