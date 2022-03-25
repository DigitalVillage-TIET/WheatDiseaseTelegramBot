import torchvision
import torch.nn as nn


def EfficientNetB0():

    Model = torchvision.models.efficientnet_b0(pretrained = True, progress = True)
    
    for param in Model.parameters():
        param.requires_grad = False
    
    Model.classifier = nn.Sequential(
        nn.Dropout(p = 0.5, inplace = True),
        nn.Linear(in_features = 1280, out_features = 7,bias = True),
        )
    return Model

class EfficientNetB0Model(nn.Module):
    def __init__(self):
        super(EfficientNetB0Model, self).__init__()
        self.baseModel = EfficientNetB0()

    def forward(self, x):
        output = self.baseModel(x)
        return output
