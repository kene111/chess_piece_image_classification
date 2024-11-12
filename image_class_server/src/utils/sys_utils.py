import torch
import torch.nn as nn

import torchvision
from torchvision import models
from torchvision import transforms


import random
import base64
import tempfile
import numpy as np
from PIL import Image
from  io import StringIO


from ..config.sys_config import  ImageClassConfig

IMAGE_SIZE = 224


def image_transforms(img):
    img_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # 
        std=[ 0.229, 0.224, 0.225] #
    )])

    img = img_transforms(img)
    return img.to(ImageClassConfig.system_device)


def get_tensor_image(image_str):
    image_bytes = bytes(image_str, encoding='utf-8')
    photo_data = base64.b64decode(image_bytes)
    data_buffer = np.frombuffer(photo_data, offset=0, dtype = np.uint8)
    data_buffer = data_buffer.astype("float32")
    img = Image.fromarray(data_buffer)
    img = img.convert('RGB')
    img_tensor = image_transforms(img)
    return img_tensor

    

def load_saved_model(path):
    pretrained_model = models.resnet50(weights='IMAGENET1K_V2')
    num_ftrs = pretrained_model.fc.in_features
    class_targets = len(ImageClassConfig.image_targets)
    
    pretrained_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, class_targets)
    )

    pretrained_model.load_state_dict(torch.load(path, weights_only=True, map_location=torch.device('cpu')))
    return pretrained_model.eval()