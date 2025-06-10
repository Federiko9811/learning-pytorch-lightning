import lightning as pl

from config import INPUT_SIZE, NUM_CLASSES, device, BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS
from dataset import MnistDataModule
from model import NN
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == "__main__":
    model = NN(INPUT_SIZE, NUM_CLASSES).to(device)

    logger = TensorBoardLogger(
        save_dir="logs",
        name="mnist_v0",
    )

    datamodule = MnistDataModule(
        data_dir="dataset/",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=1,
        min_epochs=1,
        max_epochs=NUM_EPOCHS,
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
    trainer.test(model, datamodule)
