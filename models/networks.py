import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import torch
from torch.nn import CrossEntropyLoss




class MobileNetV3_Network(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3_Network, self).__init__()
        backbone = backbone_.mobilenet_v3_large(pretrained=True) 
        # backbone_.v()

        self.features = backbone.features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(960, num_classes)
        self.num_classes = num_classes


    def forward(self, x, labels=None, last=False, freeze=False):
    
        if freeze:
            with torch.no_grad():

                e = self.features(x)
                e = self.pool_method(e)
                e = torch.flatten(e, 1)

        else:
            
            e = self.features(x)
            e = self.pool_method(e)
            e = torch.flatten(e, 1)

        
        logits = self.classifier(e)
        # logits = x[...,0,0]

        if last:
            return logits, e
        else:
            if labels is not None:
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
                return loss

            else:
                return logits

    def get_embedding_dim(self):
        return 960
            
            


