import lightning as pl
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import CIFARDataModule
from model import CIFAR10CNN

BATCH_SIZE = 64
NUM_WORKERS = 15

if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")

    # Initialize model
    model = CIFAR10CNN(learning_rate=0.001)

    # Setup logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name="cifar10_cnn",
        version=None  # Auto-increment version
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        filename='cifar10-{epoch:02d}-{val_acc:.2f}',
        save_last=True,
        verbose=True
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Setup data module
    datamodule = CIFARDataModule(
        data_dir="dataset/",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Enhanced trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor, RichProgressBar()],
        accelerator="auto",
        devices=1,
        min_epochs=5,
        max_epochs=50,
        precision="16-mixed",  # Mixed precision for faster training
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=1,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True  # For reproducibility
    )

    # Training
    trainer.fit(model, datamodule)

    # Test with best checkpoint
    trainer.test(model, datamodule, ckpt_path="best")

    print(f"\nTraining completed!")
    print(f"TensorBoard logs saved to: {logger.log_dir}")
    print(f"To view results, run: tensorboard --logdir={logger.save_dir}")
