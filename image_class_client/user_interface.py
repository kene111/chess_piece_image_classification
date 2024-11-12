import streamlit as st

import os
import sys
import time
import base64
import requests
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO


st.title("CHESS PIECE IMAGE CLASSIFICATION TASK")
uploaded_image = st.file_uploader('Kindly upload any photo of a chess piece.', type=['png', 'jpeg', "jpg"])
url_api = "http://127.0.0.1:5000/classify"

def load_image(filename):
   """Load model from filename (path)"""
    im = Image.open(filename).convert('RGB')
    im_np = np.array(im)
    im_np = Image.fromarray(im_np)
    return im_np, im


if uploaded_image is not None:
    # bytes_data = uploaded_image.getvalue()

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, uploaded_image.name)
        with open(path, "wb") as f:
            f.write(uploaded_image.getvalue())

        image, pil_img = load_image(path)
        st.image(image)
            
        b64_string = base64.b64encode(open(path, "rb").read()) #open(path, "rb").read()
        b64_string = str(b64_string, encoding='utf-8')

        req_body = {'img_b64':b64_string}
        stt = time.time()
        response = requests.post(url= url_api, json=req_body)
        et = time.time()
        elapsed_time = et - stt
        
        if  response.status_code == 200 :
            request_size = f"{sys.getsizeof(req_body) /(1<<20):,.4f} MB"
            result = response.json()
            st.success(f"Chess Piece Prediction: {result['system_response']['target']}, Confidence Score: {result['system_response']['confidence']} ")
            st.write(f"Inference Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}, Size of Request: {request_size}")
        else:
           st.warning("Something seems wrong !")
            