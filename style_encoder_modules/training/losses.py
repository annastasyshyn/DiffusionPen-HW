import torch.nn as nn


# ================ Performance and Loss Function ========================
def performance(pred, label):

    loss = nn.CrossEntropyLoss()

    loss = loss(pred, label)
    return loss
