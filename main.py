import lightning as L
import torch
from model import LightningMNIST,  PyTorchMNIST
from data import MNISTDataModule
from lightning.pytorch.loggers import TensorBoardLogger




if __name__ == '__main__':

    print('Setting up DataModule')
    dm = MNISTDataModule()

    print('Setting Up Models')
    pytorch_model = PyTorchMNIST(num_features=784, num_classes=150, hidden_dimensions=[80,50,20])
    lightning_model = LightningMNIST(model=pytorch_model, learning_rate=0.01)

    logger = TensorBoardLogger("lightning_logs", name="mnist_model")

    trainer = L.Trainer(
        max_epochs=50, accelerator="cpu", devices="auto", deterministic=True, logger=logger
    )
    trainer.fit(model=lightning_model, datamodule=dm)

    train_acc = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_acc"]
    val_acc = trainer.validate(datamodule=dm)[0]["val_acc"]
    test_acc = trainer.test(datamodule=dm)[0]["test_acc"]
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )