import torch
import lightning as L
from typing import List
import torchmetrics
import torch.nn.functional as F

class PyTorchMNIST(torch.nn.Module):
    def __init__(self, num_features:int, num_classes:int, hidden_dimensions:List):
        super().__init__()
        self.num_classes = num_classes
        self.all_layers = []
        for hidden_dimension in hidden_dimensions:
            linear = torch.nn.Linear(num_features, hidden_dimension)
            self.all_layers.append(linear)
            self.all_layers.append(torch.nn.ReLU())
            num_features = hidden_dimension

        self.all_layers.append(torch.nn.Linear(hidden_dimensions[-1], num_classes))

        self.all_layers = torch.nn.Sequential(*self.all_layers)


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits
    




class LightningMNIST(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.model.num_classes)

    def forward(self,x):
        return self.model(x)
    
    def _shared_step(self, batch):
        features, true_labels = batch

        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels
    
    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log('train_loss', loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


