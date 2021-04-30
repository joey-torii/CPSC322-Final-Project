import os
import pickle

from flask import Flask
from flask import render_template
from flask import request, jsonify, redirect

app = Flask(__name__)

########################################################
#
# TODO: FIX AND COMMENT THIS WHOLE FILE
#
########################################################

@app.route("/", methods=["GET"])
def index():
    # return content and status code
    return "<h1>Welcome to my app</h1>", 200

if __name__ == "__main__":
    app.run()