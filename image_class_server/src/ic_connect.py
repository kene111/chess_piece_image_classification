from flask import Blueprint

image_class_endpoints = Blueprint('image_class_endpoints', __name__)

import os
import json
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
from flask import jsonify, request, Response
from .utils.sys_utils import get_tensor_image
from .config.sys_config import ImageClassConfig
from .components.classification import ImageClassify


@image_class_endpoints.route('/ic_alive', methods=['GET'])
@cross_origin()
def ic_alive():
    ic_response = ImageClassConfig.alive
    return Response(response=json.dumps(ic_response), status=200, mimetype='application/json')

@image_class_endpoints.route('/classify', methods=['POST'])
@cross_origin()
def classify_image():
    ic = ImageClassify()
    img_str = request.get_json()['img_b64']
    tensor_image = get_tensor_image(img_str)
    target, conf = ic.classify(tensor_image)
    response = {"system_response": {"target":target, "confidence": conf}}
    return Response(response=json.dumps(response), status=200, mimetype='application/json')
