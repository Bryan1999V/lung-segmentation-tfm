import torch.nn as nn
import torchvision.models as models
from transformers import AutoModelForImageClassification
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_L_Weights,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    ViT_B_16_Weights,
    ViT_L_16_Weights,
    ViT_H_14_Weights,
)

CLS_N_CLASSES = 1  # Binary classification: benign vs malignant


class ClassificationModels2D:

    class _Resnet(nn.Module):
        def __init__(self, weights: str = "18"):
            super().__init__()
            if weights == "18":
                self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

            elif weights == "50":
                self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

            elif weights == "152":
                self.model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

            else:
                raise ValueError(f"Weight {weights} not recognized.")

            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_features, CLS_N_CLASSES))

        def forward(self, x):
            return self.model(x)

    class _EfficientNetV2(nn.Module):
        def __init__(self, weight_size: str = "S"):
            super().__init__()
            if weight_size == "S":
                self.model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            elif weight_size == "M":

                self.model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            elif weight_size == "L":

                self.model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Weight size {weight_size} not recognized.")

            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(nn.Linear(num_features, CLS_N_CLASSES))

        def forward(self, x):
            return self.model(x)

    class _ConvNeXt(nn.Module):
        def __init__(self, weight_size: str = "S"):
            super().__init__()
            if weight_size == "S":
                self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

            elif weight_size == "M":
                self.model = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

            elif weight_size == "L":
                self.model = models.convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Weight size {weight_size} not recognized.")

            # Replace only the final Linear layer, keep LayerNorm2d and Flatten
            num_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(num_features, CLS_N_CLASSES)

        def forward(self, x):
            return self.model(x)

    class _ViT(nn.Module):
        def __init__(self, weight_size: str = "S"):
            super().__init__()
            if weight_size == "S":
                self.model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

            elif weight_size == "M":
                self.model = models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)

            elif weight_size == "L":
                self.model = models.vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)

            else:
                raise ValueError(f"Weight size {weight_size} not recognized.")

            num_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Sequential(nn.Linear(num_features, CLS_N_CLASSES))

        def forward(self, x):
            return self.model(x)

    class _PvT(nn.Module):
        def __init__(self, weight_size: str = "S"):
            super().__init__()

            if weight_size == "S":
                model_name = "OpenGVLab/pvt_v2_b2"

            elif weight_size == "M":
                model_name = "OpenGVLab/pvt_v2_b4"

            elif weight_size == "L":
                model_name = "OpenGVLab/pvt_v2_b5"

            else:
                raise ValueError(f"Weight size {weight_size} not recognized.")

            # Load pretrained model
            self.model = AutoModelForImageClassification.from_pretrained(model_name)

            # Replace the classifier head for binary classification
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, CLS_N_CLASSES)

        def forward(self, x):
            outputs = self.model(pixel_values=x)
            return outputs.logits


    @staticmethod
    def get_model(model_name, weight_size: str):
        if "resnet" in model_name:
            return ClassificationModels2D._Resnet(weight_size)

        elif "efficientnet" in model_name:
            return ClassificationModels2D._EfficientNetV2(weight_size)

        elif "convnext" in model_name:
            return ClassificationModels2D._ConvNeXt(weight_size)

        elif "vit" in model_name:
            return ClassificationModels2D._ViT(weight_size)

        elif "pvt" in model_name:
            return ClassificationModels2D._PvT(weight_size)

        else:
            raise ValueError(f"Model {model_name} not recognized.")
