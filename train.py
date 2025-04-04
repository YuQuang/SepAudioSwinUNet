import torch
import lightning as L
from typing import Tuple
from models.swin_unet_lass import SwinUnetLASS
from utils.clotho_dataset import Clotho
from utils.esc50_dataset import ESC50
from torch.utils.data import DataLoader, random_split

def load_data() -> Tuple[DataLoader, DataLoader]:
    train_dataset, valid_dataset = random_split(
        ESC50(), [0.9, 0.1]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=4,
        persistent_workers=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=4,
        num_workers=4,
        persistent_workers=True
    )
    # train_loader = DataLoader(
    #     Clotho(ftype="development"),
    #     batch_size=5,
    #     num_workers=4,
    #     persistent_workers=True
    # )
    # valid_loader = DataLoader(
    #     Clotho(ftype="validation"),
    #     batch_size=5,
    #     num_workers=4,
    #     persistent_workers=True
    # )
    return train_loader, valid_loader

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    autoencoder = SwinUnetLASS()

    train_loader, valid_loader = load_data()
    
    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=1000,
        profiler="simple"
    )
    trainer.fit(
        autoencoder,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path="./lightning_logs/version_3/checkpoints/epoch=399-step=275277.ckpt"
    )