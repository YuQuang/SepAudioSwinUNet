import torch
import lightning as L
from typing import Tuple
from models.swin_unet_lass import SwinUnetLASS
from utils.esc50_dataset import ESC50
from torch.utils.data import DataLoader, random_split

def load_data() -> Tuple[DataLoader, DataLoader]:
    train_dataset, valid_dataset = random_split(
        ESC50(),
        [ 0.9, 0.1 ]
    )
    train_loader      = DataLoader(
        dataset       = train_dataset,
        batch_size    = 4,
        num_workers   = 4,
        persistent_workers=True
    )
    valid_loader      = DataLoader(
        dataset       = valid_dataset,
        batch_size    = 4,
        num_workers   = 4,
        persistent_workers=True
    )
    return train_loader, valid_loader

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    model = SwinUnetLASS()

    train_loader, valid_loader = load_data()
    
    trainer         = L.Trainer(
        accelerator = "auto",
        max_epochs  = 1000,
        profiler    = "simple"
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path="./lightning_logs/version_5/checkpoints/epoch=311-step=140400.ckpt"
    )
