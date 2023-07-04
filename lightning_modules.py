import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import evaluate

class PASCALModule(L.LightningModule):
    def __init__(self, model, lr=1e-3, momentum=0.9, ignore_index=0):
        super().__init__()
        self.model = model
        self.loss_module = nn.CrossEntropyLoss()
        self.lr = lr
        self.momentum = momentum
        self.iou = evaluate.load("mean_iou")
        self.ignore_index = ignore_index

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        logits = self.model.logits(inputs)
        loss = self.loss_module(logits, labels.long())
        preds = self.model(inputs)
        iou = self.iou.compute(predictions=preds, references=labels, num_labels=22, ignore_index=self.ignore_index)
        mean_iou = iou['mean_iou']
        acc = iou['mean_accuracy']

        # log accuracy and loss
        self.log("Train Pixel Accuracy", acc)
        self.log("Train Mean IoU", mean_iou)
        self.log("Train Loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
        iou = self.iou.compute(predictions=preds, references=labels, num_labels=22, ignore_index=self.ignore_index)
        mean_iou = iou['mean_iou']
        acc = iou['mean_accuracy']

        self.log("Test Pixel Accuracy", acc)
        self.log("Tets Mean IoU", mean_iou)