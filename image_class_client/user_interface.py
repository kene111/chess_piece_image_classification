import streamlit as st

import os
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
    # Shave off some pixels from the top
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
        response = requests.post(url= url_api, json={'img_b64':b64_string})

        if  response.status_code == 200 :
            result = response.json()
            st.success(f"Chess Piece Prediction: {result['system_response']}")
        else:
           st.warning("Something seems wrong !")
            