from flask import Flask, request
import numpy as np
import datetime
import os

import cv2

from face_detection import *


# Flask
app = Flask(__name__)


@app.route("/")
def Face():
    return '<h1>Face</h1>'


@app.route("/upload", methods=['POST', 'GET'])
def upload():
    face_name = request.form.get('face_name', '')
    upload_file = request.files['image']

    file_name = face_name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    upload_path = os.path.join("./Resource/tmp", file_name)
    upload_file.save(upload_path)
    print(upload_path)

    return upload_path


@app.route("/face_detect", methods=['POST', 'GET'])
def Flask_FaceDetect():
    if request.method == 'POST':
        upload_file = request.files['image']

        im_data = np.fromstring(upload_file.read(), np.uint8)
        im_data = cv2.imdecode(im_data, cv2.IMREAD_COLOR)
        # cv2.imshow('POST', im_data)
        # cv2.waitKey(0)
    else:
        im_url = request.args.get("url")
        im_data = cv2.imread(im_url)
        # cv2.imshow('URL', im_data)
        # cv2.waitKey(0)

    pos = FaceDetection(im_data)

    return str(pos)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=90, debug=True)
