import torch.nn as nn
import timm


class Mixed_Encoder(nn.Module):

    def __init__(
        self, model_name="resnet50", num_classes=339, pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool=""
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        if hasattr(self.model, "num_features"):
            num_features = self.model.num_features
        else:
            num_features = 2048

        self.projection = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(num_features, num_classes)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        features = self.model(x)
        embedding = self.global_pool(features).flatten(1)  

        projected = self.projection(embedding)  
        logits = self.classifier(projected)

        return logits, embedding
