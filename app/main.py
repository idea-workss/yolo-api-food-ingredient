"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image

import torch
from flask import Flask, request

app = Flask(__name__)

DETECTION_URL = "/api/yolov5sv1"

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).autoshape()  # force_reload = recache latest code
model.eval()


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


if __name__ == "__main__":
    app.run()  # debug=True causes Restarting with stat
