"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image

import torch
from flask import Flask, request, Response, send_from_directory

import search_engine
import json

import os
import glob
import cv2
import numpy as np

app = Flask(__name__, static_folder='FoodImages', static_url_path='/api/resource/')

@app.route('/', methods=['GET'])
def home():
    return "<h1>API</h1>"

DETECTION_URL = "/api/yolov5sv1"
@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.ones(img.shape, dtype="uint8") * 30
        img = cv2.add(img, brightness)
        #img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=400)
        data = results.pandas().xyxy[0]

        if len(data) == 0:
            return Response(json.dumps(["No Ingredients Detected"]), mimetype='application/json')
        else:
            return Response(data.to_json(orient="records"), mimetype='application/json')

SEARCH_URL = "/api/search"
@app.route(SEARCH_URL, methods=["POST"])
def search():
    if not request.method == "POST":
        return "not post"
    
    query = request.form["query"]
    result = search_engine.search(query)
    
    return Response(result.to_json(orient="records"), mimetype='application/json')

BENEFIT_URL = "/api/benefits"
@app.route(BENEFIT_URL, methods=["POST"])
def benefits():
    if not request.method == "POST":
        return "use POST"

    if request.form['fruit']:
        input_fruit = request.form['fruit']
        path_benefit = "benefits//"                                 # Path of Database
        list_txt = glob.glob(os.path.join(path_benefit, '*.txt'))   # Get all of .txt data

        # Eliminate the path to get the right .txt
        used_path = ""
        for idx, data in enumerate(list_txt):
            if data.split("/")[1].split(".")[0] == input_fruit:
                used_path = used_path + list_txt[idx]
                break

        if used_path == '':
            string = ["No Result Found"]
            
            return Response(json.dumps(string), mimetype='application/json')
        else:
            # Read .txt
            with open(used_path) as f:
                lines = f.readlines()

            # Remove enter denotation
            for idx, string in enumerate(lines):
                lines[idx] = string.replace("\n", '')

            return Response(json.dumps(lines), mimetype='application/json')
    else:
        string = ["No Request Found"]
        return Response(json.dumps(string), mimetype='application/json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('Sekigahara/yolov5-medium-custom', 'custom', path='best_yolov5m_20_epoch_12bs.pt')
    model.eval()
    
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port, debug=True)  # debug=True causes Restarting with stat
