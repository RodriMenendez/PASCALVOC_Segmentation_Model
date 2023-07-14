import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import evaluate

class PascalModule(L.LightningModule):
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        labels_one_hot = torch.permute(F.one_hot(labels.to(torch.int64), num_classes=22), (0, 3, 1, 2))

        logits = self.model.logits(inputs)
        loss = self.loss_module(logits, labels_one_hot.float())
        if batch_idx % 30 == 0:
            preds = self.model(inputs)
            iou = self.iou.compute(predictions=preds, references=labels, num_labels=22, ignore_index=self.ignore_index)
            mean_iou = iou['mean_iou']
            acc = iou['mean_accuracy']

            # log accuracy and loss
            self.log("Train Pixel Accuracy", acc)
            self.log("Train Mean IoU", mean_iou)

        self.log("Train Loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        if batch_idx % 30 == 0:
            preds = self.model(inputs)
            batch_iou = self.iou.compute(predictions=preds, references=labels, num_labels=22, ignore_index=self.ignore_index)
            mean_iou = batch_iou['mean_iou']
            acc = batch_iou['mean_accuracy']

            # log accuracy and loss
            self.log("Val Pixel Accuracy", acc)
            self.log("Val Mean IoU", mean_iou)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
        iou = self.iou.compute(predictions=preds, references=labels, num_labels=22, ignore_index=self.ignore_index)
        mean_iou = iou['mean_iou']
        acc = iou['mean_accuracy']

        self.log("Test Pixel Accuracy", acc)
        self.log("Test Mean IoU", mean_iou)