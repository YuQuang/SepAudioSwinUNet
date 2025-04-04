import torch
import torchaudio
import lightning as L
import torch.nn as nn
from torch import optim
from models.stft import STFT
from models.swin_unet.vision_transformer import SwinUnet
from models.bert.word_embbeding import WordEmbbeding
from transformers import HubertModel
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


class RelativePositionEncoding(nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(length, dim))  # 可學習的位置嵌入

    def forward(self, x):
        return x + self.pos_emb[:x.shape[1], :].unsqueeze(0)  # 加到輸入上


class SwinUnetLASS(L.LightningModule):
    def __init__(
            self,
            embed_dim   = 512,
            n_fft       = 1023,
            hop_length  = 1023 // 4,
            win_length  = 1023
        ):
        super().__init__()
        
        # Word Embbeding
        self.word_emb   = nn.Sequential(
            WordEmbbeding(),
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        # STFT Module
        self.stft       = STFT(
            n_fft       = n_fft,
            hop_length  = hop_length,
            win_length  = win_length
        )
        # Cross Attention
        self.pos_enc    = RelativePositionEncoding(embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = 8,
            dropout     = 0.1,
            batch_first = True
        )
        # Mask Generator
        self.generator  = nn.Sequential(
            SwinUnet(img_size=embed_dim, num_classes=4),
            nn.ReLU()
        )
        
    def forward(self, audio: torch.Tensor, query: list[str], output_mag: bool = False):
        #
        # STFT Encoder
        #
        mag, pha, L  = self.stft.stft(audio)                        # STFT

        #
        # Text Query 與 STFT Magnitude 進行 Cross Attention
        # 1. Text Query 進行 BERT Embbeding
        # 2. STFT Magnitude 進行 Cross Attention
        # 3. Cross Attention 的輸出計算Mask
        # 4. Mask 與 STFT Magnitude 哈達瑪乘積
        #
        query        = self.word_emb(query)                         # Query (B, tokens, 768)
        residual     = mag
        mag_query    = self.pos_enc(mag.squeeze(1).transpose(1,2))
        mag_query, _ = self.cross_attn(                             # CrossAttention Query & Magnitude
                            query = mag_query,
                            key   = query,
                            value = query
                        )
        mag_query    = residual + 0.5 * mag_query.transpose(1,2).unsqueeze(1)
        masked_mag   = self.generator(mag_query) * mag              # Apply Mask

        #
        # ISTFT Decoder
        #
        out          = self.stft.istft(                             # ISTFT
                            torch.cat((masked_mag, pha), dim=1),
                            L
                        )
        if output_mag: return out, masked_mag
        return out

    def training_step(self, batch, batch_idx):
        audio, query, target = batch
        pred, masked_mag     = self.forward(audio, query, output_mag=True)
        loss                 = our_loss(
                                    pred, target,
                                    ( masked_mag, self.stft.stft(target)[0] )
                                )
        self.log_dict({ "train_our_loss": loss }, prog_bar=True, batch_size=audio.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        audio, query, target = batch
        pred                 = self.forward(audio, query)
        loss                 = scale_invariant_signal_noise_ratio( pred, target ).mean()
        self.log_dict({ "valid_sisnr_loss": loss }, prog_bar=True, batch_size=audio.shape[0])

    def test_step(self, batch, batch_idx):
        audio, query, target = batch
        pred     = self.forward(audio, query)

        loss = scale_invariant_signal_noise_ratio(pred, target).mean()
        self.log_dict({ "test_sisnr_loss": loss }, prog_bar=True, batch_size=audio.shape[0])

        pred     = pred.cpu().detach()[0]
        audio    = audio.cpu().detach()[0]
        target   = target.cpu().detach()[0]
        # torchaudio.save("./temp/out.wav", pred, 32000)
        # torchaudio.save("./temp/mix.wav", audio, 32000)
        # torchaudio.save("./temp/target.wav", target, 32000)

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {
                "params": self.parameters(),
                "lr": 3e-4,
                "weight_decay": 0.01
            }   # SeparationNET
        ])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
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

def our_loss(
        pred: torch.Tensor, tgt: torch.Tensor,
        mag: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
    sisnr_loss     = scale_invariant_signal_noise_ratio(pred, tgt).negative().mean()
    mag_l1_loss    = torch.nn.functional.smooth_l1_loss(*mag).mean()
    spec_l1_loss   = torch.nn.functional.l1_loss(pred, tgt).mean()
    return sisnr_loss + 0.5 * mag_l1_loss + 0.5 * spec_l1_loss
