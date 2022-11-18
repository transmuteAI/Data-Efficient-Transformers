# '''
# ViT in PyTorch.
# '''


# import torch.nn as nn
# # import torch.nn.functional as F
# import torch
# import torchvision.models as models

# class ViT_B16(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ViT_B16, self).__init__()

#         self.embDim = 768

#         backbone = models.vit_b_16(pretrained=False) 
#         self.conv_proj = backbone.conv_proj
#         self.encoder = backbone.encoder
#         self.classifier = nn.Linear(768, num_classes)

#     def forward(self, x, last=False, freeze=False):
#         if freeze:
#             with torch.no_grad():

#                 out = self.conv_proj(x)
#                 out = self.encoder(out)
#                 e = out.view(out.size(0), -1)
#         else:
#             out = self.conv_proj(x)
#             out = self.encoder(out)
#             e = out.view(out.size(0), -1)
#         out = self.classifier(e)
#         if last:
#             return out, e
#         else:
#             return out

#     def get_embedding_dim(self):
#         return self.embDim

# from collections import OrderedDict
import torch
import torch.nn as nn
# import torchvision.models import vision_transformer
import torchvision.models as models


# class Vit_B16(vision_transformer.VisionTransformer):
    
#     def __init__(self, image_size: int = 32,
#         patch_size: int =16,
#         num_layers: int =12,
#         num_heads: int =12,
#         hidden_dim: int = 768,
#         mlp_dim: int = 3072,
#         dropout: float = 0.0,
#         attention_dropout: float = 0.0,
#         num_classes: int = 1000,
#         no_grad=False, **kwargs):
        
#         super(Vit_B16, self).__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, **kwargs)
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.hidden_dim = hidden_dim  
#         self.mlp_dim = mlp_dim
#         self.no_grad = no_grad
#         self.embDim = 768

#         # self.embedding_recorder = EmbeddingRecorder(record_embedding)
#         # self.heads = models.vit_b_16().heads
#         self.heads = nn.Linear(768, num_classes)
#         # nn.Sequential(OrderedDict([
#         #   ('head', ),
#         # ]))

#     def forward(self, x, last=False, freeze=False):

#         if freeze:
#             with torch.no_grad():

#                 # Reshape and permute the input tensor
#                 x = self._process_input(x)
#                 n = x.shape[0]

#                 # Expand the class token to the full batch
#                 batch_class_token = self.class_token.expand(n, -1, -1)
#                 x = torch.cat([batch_class_token, x], dim=1)

#                 x = self.encoder(x)

#                 # Classifier "token" as used by standard language architectures
#                 x = x[:, 0]
#         else:
#             # Reshape and permute the input tensor
#             x = self._process_input(x)
#             n = x.shape[0]

#             # Expand the class token to the full batch
#             batch_class_token = self.class_token.expand(n, -1, -1)
#             x = torch.cat([batch_class_token, x], dim=1)

#             x = self.encoder(x)

#             # Classifier "token" as used by standard language architectures
#             x = x[:, 0]

#         out = self.heads(x)

#         if last:
#             return out, x
#         else:
#             return out


#     def get_embedding_dim(self):
#         return self.embDim

class Vit_B16(nn.Module):
    def __init__(self, num_classes=10):
        super(Vit_B16, self).__init__()
        
        self.embDim = 768

        backbone = models.vit_b_16(pretrained=False)
        # backbone.load_state_dict(torch.load('/workspace/cords2/pretrained_weights/ViT-B_16.npz')) 

        self.image_size = backbone.image_size
        self.patch_size = backbone.patch_size
        self.class_token = backbone.class_token
        self.hidden_dim = backbone.hidden_dim

        self.conv_proj = backbone.conv_proj
        self.encoder = backbone.encoder
        self.classifier = nn.Linear(768, num_classes)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x, last=False, freeze=False):
        
        if freeze:
            with torch.no_grad():
                x = self._process_input(x)
                n = x.shape[0]

                # Expand the class token to the full batch
                batch_class_token = self.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)

                x = self.encoder(x)

                # Classifier "token" as used by standard language architectures
                e = x[:, 0]

        else:
            
            # out = self.features(x)
            # out = self.pool_method(out)
            # e = out.view(out.size(0), -1)
            x = self._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.encoder(x)

            # Classifier "token" as used by standard language architectures
            e = x[:, 0]

        out = self.classifier(e)
        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim




# def vit_b_16(image_size, patch_size, num_layers,  num_heads, hidden_dim, mlp_dim, numclasses, no_grad):

#     return Vit_B16(image_size=image_size, patch_size=patch_size, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim, num_classes=numclasses, no_grad=no_grad)

# if __name__ == '__main__':
#     vitb16 = Vit_B16()
#     xx = torch.randn(1, 3, 224, 224)
#     out = vitb16(xx)
#     print(vitb16)
 
        
# def Vit(arch: str, channel: int, num_classes: int, im_size, patch_size, record_embedding: bool = False, no_grad: bool = False,
#         pretrained: bool = False):
#     arch = arch.lower()
#     if pretrained:
#         if im_size[0] != 224 or im_size[1] != 224:
#             raise NotImplementedError("torchvison pretrained models only accept inputs with size of 224*224")
#         # net = VGG_224x224(features=vgg.make_layers(cfg_vgg[arch], True), channel=3, num_classes=1000,
#         #                   record_embedding=record_embedding, no_grad=no_grad)

#         from torch.hub import load_state_dict_from_url
#         # state_dict = load_state_dict_from_url(vgg.model_urls[arch], progress=True)
#         # net.load_state_dict(state_dict)

#         if channel != 3:
#             net.features[0] = nn.Conv2d(channel, 64, kernel_size=3, padding=1)

#         if num_classes != 1000:
#             net.fc = nn.Linear(4096, num_classes)
#             net.classifier[-1] = net.fc

#     elif im_size[0] == 224 and im_size[1] == 224:
#         net = Vit_b_16(image_size=im_size[0], patch_size=16,
#         num_layers=12,
#         num_heads=12,
#         hidden_dim=768,
#         mlp_dim=3072, num_classes=num_classes, record_embedding=record_embedding, no_grad=no_grad)

#     elif (channel == 1 and im_size[0] == 28 and im_size[1] == 28) or (
#             channel == 3 and im_size[0] == 32 and im_size[1] == 32):
#         net = Vit_b_16(image_size=im_size[0], patch_size=16,
#         num_layers=12,
#         num_heads=12,
#         hidden_dim=768,
#         mlp_dim=3072, num_classes=num_classes, record_embedding=record_embedding, no_grad=no_grad)
#     else: 
#         raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    
#     print(net)
#     return net
# def VGG13(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
#           pretrained: bool = False):
#     return VGG('vgg13', channel, num_classes, im_size, record_embedding, no_grad, pretrained)


# def VGG16(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
#           pretrained: bool = False):
#     return VGG('vgg16', channel, num_classes, im_size, record_embedding, no_grad, pretrained)


# def VGG19(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
#           pretrained: bool = False):
#     return VGG('vgg19', channel, num_classes, im_size, record_embedding, no_grad, pretrained)

# if __name__ == "__main__":
#     # net = VGG_224x224(features=vgg.make_layers(cfg_vgg["vgg11"], True), channel=3, num_classes=1000,
#     net = Vit(image_size=(32, 32), patch_size=(8, 8), num_classes=10, record_embedding=True, no_grad=False)
