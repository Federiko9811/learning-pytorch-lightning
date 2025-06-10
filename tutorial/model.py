import lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn, optim


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _common_step(self, batch):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_function(scores, y)
        return loss, scores, y

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
            {
                "training_loss": loss,
                "training_acc": accuracy,
                "training_f1": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        predictions = torch.argmax(scores, dim=1)
        return predictions

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
