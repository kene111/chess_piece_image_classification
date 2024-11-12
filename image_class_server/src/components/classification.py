import torch
import torch.nn.functional as F
from ..config.sys_config import ImageClassConfig

class ImageClassify:
    """Image classifier object"""
    def __init__(self):
        self.model = ImageClassConfig.classification_model
        self.classes = ImageClassConfig.image_targets

    def classify(self, img):
        """classify tensor image"""
        img = img.to(ImageClassConfig.system_device)
        img.unsqueeze(0)
        outputs = self.model(img.unsqueeze(0))
        outputs = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(outputs, 1)
        target = self.classes[predicted.item()]
        conf = conf.item()
        conf = f"{conf:.2f}"
        return str(target), conf