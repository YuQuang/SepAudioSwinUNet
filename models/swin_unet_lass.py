import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from models.stft import STFT
from models.swin_unet.vision_transformer import SwinUnet
from models.bert.word_embbeding import WordEmbbeding
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


class RelativePositionEncoding(nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(length, dim))  # 可學習的位置嵌入

    def forward(self, x):
        return x + self.pos_emb[:x.shape[1], :].unsqueeze(0)  # 加到輸入上


class LightTextGuidedMaskFusion(nn.Module):
    def __init__(self, embed_dim=512, num_masks=3):
        super().__init__()
        self.score_head = nn.Linear(embed_dim, num_masks)  # 輕量映射出 N 個權重

    def forward(self, text_embed, masks):
        """
        text_embed: [B, L, D]
        masks: [B, N, H, W]
        """
        B, N, H, W = masks.shape
        _, L, D = text_embed.shape

        # Step 1: 平均詞向量 → [B, D]
        text_vec = text_embed.mean(dim=1)  # [B, D]

        # Step 2: 線性投影出權重分數
        attn_scores = self.score_head(text_vec)  # [B, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N]

        # Step 3: 加權融合
        weighted_mask = (attn_weights.view(B, N, 1, 1) * masks).sum(dim=1)  # [B, H, W]

        return weighted_mask.unsqueeze(1)  # [B, 1, H, W]


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
            nn.Conv2d(
                in_channels  = 1,
                out_channels = 3,
                kernel_size  = 4,
                stride       = 2,
                padding      = 1,
            ),
            nn.BatchNorm2d(3),
            SwinUnet(
                img_size=embed_dim//2,
                num_classes=3
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels  = 3,
                out_channels = 3,
                kernel_size  = 4,
                stride       = 2,
                padding      = 1
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        # Mask Query
        self.TGMF       = LightTextGuidedMaskFusion(
            embed_dim=embed_dim,
            num_masks=3
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
        query        = self.word_emb(query)

        residual     = mag
        mag_query    = self.pos_enc(mag.squeeze(1).transpose(1,2))
        mag_query, _ = self.cross_attn(                             # CrossAttention Query & Magnitude
                            query = mag_query,
                            key   = query,
                            value = query
                        )
        mag_query    = residual + 0.5 * mag_query.transpose(1,2).unsqueeze(1)
        masks        = self.generator(mag_query)
        masked_mag   = self.TGMF(query, masks) * mag                # Apply Mask

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
        
        torchaudio.save("./temp/out.wav", pred, 32000)
        torchaudio.save("./temp/mix.wav", audio, 32000)
        torchaudio.save("./temp/target.wav", target, 32000)

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
