import torch
import torch.nn as nn

class STFT(nn.Module):
    def __init__(
            self,
            n_fft: int,
            hop_length: int,
            win_length: int
        ):
        super().__init__()
        self.n_fft      = n_fft                          # FFT 長度
        self.hop_length = hop_length                     # 步長 (10ms)
        self.win_length = win_length                     # 窗口長度
        self.window     = torch.hann_window(win_length)  # 使用 Hann 窗函數
    
    def stft(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        self.window = self.window.to(x.device)
        B, L = x.shape[0], x.shape[2]
        out = x.view(B, -1)
        out = torch.stft(
            out,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        magnitude = out.abs().unsqueeze(1)
        phase     = out.angle().unsqueeze(1)

        return magnitude, phase, L

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor):
        assert x.shape == mask.shape, "stft and mask must have the same shape"

        x_magnitude     = x[:, :(self.n_fft+1)//2, :] * mask[:, :(self.n_fft+1)//2, :]
        x_phase         = x[:, (self.n_fft+1)//2:, :] + mask[:, (self.n_fft+1)//2:, :]

        return torch.cat([ x_magnitude, x_phase ], dim=1)

    def istft(self, x: torch.Tensor, length: int) -> torch.Tensor:
        self.window = self.window.to(x.device)
        magnitude, phase = x[:, 0, ...], x[:, 1, ...]
        out = torch.istft(
            magnitude * torch.exp(1j*phase),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length
        )
        return out.view(x.shape[0], 1, -1) # out: Batch, channel, Length

