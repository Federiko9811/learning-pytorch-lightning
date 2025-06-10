import lightning as pl
from lightning.pytorch.callbacks import Callback


class PrintCallback(Callback):

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        print("Start Training")

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        print("End Training")
