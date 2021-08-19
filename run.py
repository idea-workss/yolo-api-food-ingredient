import argparse
import io
from PIL import Image

import torch
from flask import Flask, request

my_app = Flask(__name__)

@my_app.route('/')
def hello_world():
    return 'Hello World!'

DETECTION_URL = "/api/yolov5sv1"
@my_app.route(DETECTION_URL, methods=["POST"])
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




if __name__ == '__main__':
    my_app.run()
