import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import lightning as L
from lightning.pytorch.loggers import WandbLogger

import get_data
import models
import lightning_modules

import time


# get the data
train_data, trainloader, test_data, testloader = get_data.get_PASCAL()

# get models
resnet_model = models.ResNet18()
seg_model = models.Segmentation(resnet_model, out_channels=64)

# wandb
wandb_config = {
        "dataset": "PASCALVOC",
        "architecture": "Resnet18+FPN"
    }

wandb_logger = WandbLogger(project='PASCALVOC-Resnet18-Segmentation', config=wandb_config)

# lightning module
autoencoder = lightning_modules.PASCALModule(seg_model)

# train
trainer = L.pytorch.Trainer(accelerator='auto', max_epochs=4, logger=wandb_logger)
trainer.fit(model=autoencoder, train_dataloaders=trainloader)

# test
test_result = trainer.test(autoencoder, dataloaders=testloader, verbose=False)

# save model
timestr = time.strftime("%Y%m%d-%H%M%S")

path = './seg_model' + timestr
torch.save(seg_model.state_dict(), path)