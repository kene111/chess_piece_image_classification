import torch
from ..config.sys_config import ImageClassConfig

class ImageClassify:
    
    def __init__(self):
        self.model = ImageClassConfig.classification_model
        self.classes = ImageClassConfig.image_targets

    def classify(self, img):
        print(img.shape)
        img = img.to(ImageClassConfig.system_device)
        img.unsqueeze(0)
        outputs = self.model(img.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
        target = self.classes[predicted.item()]
        return target