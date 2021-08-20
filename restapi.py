"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image

import torch
from flask import Flask, request

import search_engine
import json

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "<h1>API</h1>"

DETECTION_URL = "/api/yolovsv1"
@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=640)
        data = results.pandas().xyxy[0].to_json(orient="records")
        return data

SEARCH_URL = "/api/search"
@app.route(SEARCH_URL, methods=["POST"])
def search():
    if not request.method == "POST":
        return "not post"
    
    query = request.form["query"]
    result = search_engine.search(query)
    
    return result.to_json(orient="split")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    model.eval()
    app.run()  # debug=True causes Restarting with stat
