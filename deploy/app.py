from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pydicom
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model={}
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

def get_dicom_field_value(val):

    if type(val) == pydicom.multival.MultiValue:
        return int(val[0])
    else:
        return int(val)

def get_windowing(data):

    dicom_fields = [data.WindowCenter, data.WindowWidth, data.RescaleSlope, data.RescaleIntercept]
    return [get_dicom_field_value(x) for x in dicom_fields]


def get_windowed_image(image, wc, ww, slope, intercept):

    img = (image*slope + intercept)
    img_min = wc - ww//2
    img_max = wc + ww//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img


def _normalize(img):
    if img.max() == img.min():
        return np.zeros(img.shape)
    return 2 * (img - img.min())/(img.max() - img.min()) - 1

def _read(img_path, desired_size=(224, 224)):

    # 1. read dicom file
    dcm = pydicom.dcmread(img_path)

    # 2. Extract meta data features
    # window center, window width, slope, intercept
    window_params = get_windowing(dcm)

    try:
        # 3. Generate windowed image
        img = get_windowed_image(dcm.pixel_array, *window_params)
    except:
        img = np.zeros(desired_size)

    img = _normalize(img)

    if desired_size != (512, 512):
        # resize image
        img = cv2.resize(img, desired_size, interpolation = cv2.INTER_LINEAR)
    return img[:,:,np.newaxis]

def output(value):
    for key in value:
        print (key, ' : ', value[key])



def model_predict(img_path, model):
    img = _read(img_path, desired_size=(224, 224))
    # Get the right shape
    X = np.empty((1, 224, 224, 3))
    X[0,] = img
    preds = model.predict(X)
    return preds

def convertTuple(tup):
    str =  ''.join(tup)
    return str

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        types = ["Any Hemorrhage", "Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural"]
        values = [format(x, "10.2f") for x in preds.tolist()[0]]
        result = dict(zip(types,values))
        return "".join(format(x, "10.2f") for x in preds.tolist()[0])

        #"".join(format(x, "10.2f") for x in preds.tolist()[0])

    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
