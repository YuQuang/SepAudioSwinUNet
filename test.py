import glob
import torch
import lightning as L
from models.swin_unet_lass import SwinUnetLASS
from utils.esc50_dataset import ESC50
from torch.utils.data import DataLoader

def load_data() -> DataLoader:
    test_loader     = DataLoader(
        dataset     = ESC50(),
        batch_size  = 5,
        num_workers = 4,
        persistent_workers=True
    )
    return test_loader

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    model = SwinUnetLASS.load_from_checkpoint(glob.glob("lightning_logs/version_6/**/*.ckpt")[0])
    model.eval()

    trainer         = L.Trainer(
        accelerator = "auto",
        max_epochs  = 1,
        logger      = False
    )
    trainer.test(
        model,
        dataloaders=load_data()
    )

        
