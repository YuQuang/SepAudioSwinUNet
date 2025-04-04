import torch
import torchaudio
import lightning as L
from torch import optim
from models.separation_net import SeparationNet
from models.word_embbeding import WordEmbbeding
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


class ConfLASS(L.LightningModule):
    def __init__(
            self
        ):
        super().__init__()
        self.num_sources     = 2
        self.enc_num_feats   = 768
        self.enc_kernel_size = 32
        self.enc_stride      = self.enc_kernel_size // 2
        
        #
        # 將 Query Sentence Embbeding
        #
        self.word_emb  = WordEmbbeding()
        #
        # 編碼器在時域下提取特徵
        #
        self.encoder     = torch.nn.Conv1d(
            in_channels  = 1,
            out_channels = self.enc_num_feats,
            kernel_size  = self.enc_kernel_size,
            stride       = self.enc_stride,
            padding      = self.enc_stride,
            bias         = False
        )
        #
        # 產生遮罩
        #
        self.separation_net = SeparationNet(
            feature_size    = self.enc_num_feats,
            n_head          = 2,
            n_layer         = 2
        )
        #
        # 解碼器還原分離的音訊
        #
        self.decoder      = torch.nn.ConvTranspose1d(
            in_channels   = self.enc_num_feats,
            out_channels  = 1,
            kernel_size   = self.enc_kernel_size,
            stride        = self.enc_stride,
            padding       = self.enc_stride,
            bias          = False
        )

    def forward(self, audio: torch.Tensor, query: list[str], output_mask: bool = False):
        feats, padd = self._align_num_frames_with_strides(audio)
        feats       = self.encoder(feats)            # Encoder
        mask        = self.separation_net(           # Get Mask
                          feats,
                          self.word_emb(query)       # Get Query Embbeding
                      )
        masked      = mask * feats                   # Apply Mask
        out         = self.decoder(masked)           # Decoder
        if padd > 0: out = out[..., :-padd]          # B, S, L

        if output_mask: return out, mask
        return out

    def training_step(self, batch, batch_idx):
        audio, query, target = batch
        pred, mask           = self.forward(audio, query, output_mask=True)
        loss                 = sisnr_loss(pred, target, mask)
        self.log_dict({ "train_loss": loss }, prog_bar=True, batch_size=audio.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        audio, query, target = batch
        pred, mask           = self.forward(audio, query, output_mask=True)
        loss                 = sisnr_loss(pred, target, mask)
        self.log_dict({ "valid_loss": loss }, prog_bar=True, batch_size=audio.shape[0])

    def test_step(self, batch, batch_idx):
        audio, query, target = batch
        pred     = self.forward(audio, query)

        loss = scale_invariant_signal_noise_ratio(pred, target).mean()
        print(f"Test Loss: {loss.item()}")

        pred     = pred.cpu().detach()[0]
        audio    = audio.cpu().detach()[0]
        target   = target.cpu().detach()[0]

        torchaudio.save("./temp/out.wav", pred, 32000)
        torchaudio.save("./temp/mix.wav", audio, 32000)
        torchaudio.save("./temp/target.wav", target, 32000)

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {
                "params": self.parameters(),
                "lr": 1e-4,
                "weight_decay": 0.01
            }   # SeparationNET
        ])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.9
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",  # 監控 validation loss
                "interval": "epoch",      # 每個 epoch 檢查一次
                "frequency": 1            # 每個 epoch 檢查
            }
        }

    def _align_num_frames_with_strides(self, input: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad input Tensor so that the end of the input tensor corresponds with

        1. (if kernel size is odd) the center of the last convolution kernel
        or 2. (if kernel size is even) the end of the first half of the last convolution kernel

        Assumption:
            The resulting Tensor will be padded with the size of stride (== kernel_width // 2)
            on the both ends in Conv1D

        |<--- k_1 --->|
        |      |            |<-- k_n-1 -->|
        |      |                  |  |<--- k_n --->|
        |      |                  |         |      |
        |      |                  |         |      |
        |      v                  v         v      |
        |<---->|<--- input signal --->|<--->|<---->|
         stride                         PAD  stride

        Args:
            input (torch.Tensor): 3D Tensor with shape (batch_size, channels==1, frames)

        Returns:
            Tensor: Padded Tensor
            int: Number of paddings performed
        """
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0

        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(
            batch_size,
            num_channels,
            num_paddings,
            dtype=input.dtype,
            device=input.device,
        )
        return torch.cat([input, pad], 2), num_paddings


def sisnr_loss(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sisnr_loss = scale_invariant_signal_noise_ratio(pred, tgt).negative().mean()
    mask_loss  = (mask - mask.pow(2)).abs().mean()

    B, C, L = pred.shape[0], pred.shape[1], pred.shape[2]
    window = torch.windows.hann(1024).to(pred.device)
    pred = torch.stft(
        pred.view(B, L),
        n_fft=1024,
        hop_length=1024 // 4,
        win_length=1024,
        window=window,
        return_complex=True
    )
    tgt = torch.stft(
        tgt.view(B, L),
        n_fft=1024,
        hop_length=1024 // 4,
        win_length=1024,
        window=window,
        return_complex=True
    )
    epsilon = 1e-6
    stft_loss     = torch.nn.functional.smooth_l1_loss(
                        pred.abs(),
                        tgt.abs()
                    ).mean()
    stft_log_loss = torch.nn.functional.smooth_l1_loss(
                        torch.log(pred.abs() + epsilon),
                        torch.log(tgt.abs() + epsilon)
                    ).mean()
    return sisnr_loss + 0.2 * mask_loss + 0.15 * stft_loss + 0.15 * stft_log_loss

