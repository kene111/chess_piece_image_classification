import os
from pathlib import Path
from dotenv import load_dotenv

import torch

load_dotenv()

class ImageClassConfig:
    """ IMAGE CLASSIFICATION CONFIGURATION OBJECT"""

    object_storage = "storage"

    classification_model = None
    image_targets = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']


    model_name = "best_model_epoch_22--val_acc_0.52.pth"
    model_dir = "models"
    model_path = os.path.join(model_dir, model_name)
    alive = {"system_response":"System is running!"}

    system_device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


