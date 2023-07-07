import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import evaluate

class PascalModule(L.LightningModule):
    def __init__(self, model, lr=1e-3, momentum=0.9, ignore_index=0, wd=1e-3):
        super().__init__()
        self.model = model
        weights = torch.tensor([0.5, 1, 1, 1, 1, 1, 1,\
                                1, 1, 1, 1, 1, 1, 1,\
                                1, 1, 1, 1, 1, 1, 1, 1])
        self.seg_loss_module = nn.CrossEntropyLoss()
        self.clf_loss_module = nn.CrossEntropyLoss(weight=weights)
        self.lr = lr
        self.momentum = momentum
        self.wd = wd
        self.iou = evaluate.load("mean_iou")
        self.ignore_index = ignore_index

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        return optimizer

    def labels_clf_one_hot(self, labels):
        u = None
        for label in labels:
            loh = torch.sum(F.one_hot(torch.unique(label.to(torch.int64)), num_classes=22), dim=0)
            if torch.is_tensor(u):
                try:
                    u = torch.stack((u, loh), dim=0)
                except:
                    u = torch.cat((u, loh.unsqueeze(0)))
            else:
                u = loh

        return u

    def model_classes(self, labels_clf_one_hot):
        input = labels_clf_one_hot
        input = (input==1).nonzero()

        cat_dict = {}

        for pair in input:
            pair = pair.numpy()
            index, value = pair[0], pair[1]
            if index in cat_dict:
                cat_dict[index].append(value)
            else:
                cat_dict[index] = [value]

        return cat_dict

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        # one hot encoding on labels
        labels_one_hot = torch.permute(F.one_hot(labels.to(torch.int64), num_classes=22), (0, 3, 1, 2))
        labels_clf_one_hot = self.labels_clf_one_hot(labels)

        if len(labels_clf_one_hot.shape) == 1:
            # batch dimension not on labels if only one input
            labels_clf_one_hot = labels_clf_one_hot.unsqueeze(0)

        # logits for segmentation and classification
        logits = self.model.logits(inputs)
        logits_clf = self.model.saved_classification_logits()

        # calcuating loss
        loss_seg = self.seg_loss_module(logits, labels_one_hot.float())
        loss_clf = self.clf_loss_module(logits_clf, labels_clf_one_hot.float())
        loss = torch.add(loss_seg, loss_clf)

        # calculating sementation metrics
        preds = self.model(inputs)
        iou = self.iou.compute(predictions=preds, references=labels, num_labels=22, ignore_index=self.ignore_index)
        mean_iou = iou['mean_iou']
        acc = iou['mean_accuracy']

        # calculating classification metrics

        # log accuracy and loss
        self.log("Train Pixel Accuracy", acc)
        self.log("Train Mean IoU", mean_iou)
        self.log("Train Loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)

        logits_clf = self.model.classification_logits(inputs)
        iou = self.iou.compute(predictions=preds, references=labels, num_labels=22, ignore_index=self.ignore_index)
        mean_iou = iou['mean_iou']
        acc = iou['mean_accuracy']

        # log accuracy and loss
        self.log("Validation Pixel Accuracy", acc)
        self.log("Validation Mean IoU", mean_iou)


    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
        iou = self.iou.compute(predictions=preds, references=labels, num_labels=22, ignore_index=self.ignore_index)
        mean_iou = iou['mean_iou']
        acc = iou['mean_accuracy']

        self.log("Test Pixel Accuracy", acc)
        self.log("Test Mean IoU", mean_iou)