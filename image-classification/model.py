import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import Accuracy, ConfusionMatrix


class CIFAR10CNN(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Define the CNN architecture
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes for CIFAR-10

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Define metrics for tracking accuracy
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

        # Confusion matrix for test evaluation
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=10)

        # Store learning rate
        self.learning_rate = learning_rate

        # CIFAR-10 class names for visualization
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        # This defines what happens in one training step
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        # Log histograms every 100 steps
        if batch_idx % 100 == 0:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.logger.experiment.add_histogram(f"weights/{name}", param, self.global_step)
                    self.logger.experiment.add_histogram(f"gradients/{name}", param.grad, self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        # This defines what happens in one validation step
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # This defines what happens in one test step
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y)

        # Update confusion matrix
        self.confusion_matrix.update(preds, y)

        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return loss

    def on_validation_epoch_end(self):
        # Log sample images with predictions every 5 epochs
        if self.current_epoch % 5 == 0:
            try:
                # Get validation dataloader
                val_dataloader = self.trainer.datamodule.val_dataloader()

                # Get a batch of validation data
                batch = next(iter(val_dataloader))
                images, labels = batch
                images = images[:8]  # Take first 8 images
                labels = labels[:8]

                # Move to device
                images = images.to(self.device)

                # Get predictions
                with torch.no_grad():
                    logits = self(images)
                    preds = torch.argmax(logits, dim=1)

                # Denormalize images for visualization
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(self.device)
                std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(self.device)
                images_denorm = images * std + mean
                images_denorm = torch.clamp(images_denorm, 0, 1)

                # Create grid
                grid = torchvision.utils.make_grid(images_denorm, nrow=4, normalize=False)

                # Add text annotations
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(grid.cpu().permute(1, 2, 0))
                ax.axis('off')

                # Add predictions as title
                title = "Predictions: " + " | ".join([
                    f"{self.class_names[pred]} ({'✓' if pred == label else '✗'})"
                    for pred, label in zip(preds.cpu(), labels)
                ])
                ax.set_title(title, fontsize=10)

                # Log to tensorboard
                self.logger.experiment.add_figure("sample_predictions", fig, self.current_epoch)
                plt.close(fig)

            except Exception as e:
                print(f"Could not log sample images: {e}")

    def on_test_epoch_end(self):
        # Compute and log confusion matrix
        cm = self.confusion_matrix.compute()

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm.cpu().numpy(), annot=True, fmt='d', ax=ax,
                    xticklabels=self.class_names, yticklabels=self.class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Log to tensorboard
        self.logger.experiment.add_figure("confusion_matrix", fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_start(self):
        """Log model graph when training starts"""
        try:
            # Get a sample from the training dataloader
            sample_batch = next(iter(self.trainer.datamodule.train_dataloader()))
            sample_input = sample_batch[0][:1]  # Take just one sample

            # Move to same device as model
            sample_input = sample_input.to(self.device)

            # Log the model graph
            self.logger.experiment.add_graph(self, sample_input)
            print("Model graph logged to TensorBoard")

        except Exception as e:
            print(f"Could not log model graph: {e}")
