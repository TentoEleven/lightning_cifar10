import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks.lr_finder import LRFinder
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        self.accuracy = Accuracy('multiclass', num_classes=10)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            "scheduler": OneCycleLR(optimizer, max_lr=0.01, epochs=self.trainer.max_epochs,
                                    steps_per_epoch=len(self.train_dataloader)),
            "interval": "step"  # Adjust learning rate per step
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.compute_loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log_dict({"train_loss": loss, "train_acc": acc},
                      on_step=False, on_epoch=True,
                      prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.compute_loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log_dict({"val_loss": loss, "val_acc": acc},
                      on_step=False, on_epoch=True,
                      prog_bar=True, logger=True)

# Example usage:
if __name__ == "__main__":
    # Instantiate your model and data loaders
    model = YourCustomModel()  # Replace with your actual model
    lit_model = LitModel(model)

    # Run the LR finder
    trainer = pl.Trainer(auto_lr_find=True)
    trainer.tune(lit_model)
    lr_finder = LRFinder(lit_model)
    lr_finder.range_test(train_dataloader)  # Adjust DataLoader as needed
    lr_finder.plot()  # Plot the LR range
