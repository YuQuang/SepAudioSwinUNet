import os
import glob
import random
import torch
import librosa
import pandas as pd
import pyloudnorm as pyln
import torch.nn.functional as F


def pad_waveform(
        waveform: torch.Tensor,
        max_seconds: float,
        sample_rate: int
    ) -> torch.Tensor:
    """
    Args:
        waveform:       torch.Tensor, shape ( 1, T )
        max_seconds:    float, maximum seconds to pad
        sample_rate:    int, sample rate of waveform
    Returns:
        padded_tensor:  torch.Tensor, shape ( 1, max_seconds*sample_rate )
    """
    max_length = int(max_seconds * sample_rate)
    if waveform.size(-1) < max_length:
        pad_length = max_length - waveform.size(-1)
        padded_tensor = F.pad(
            waveform[0, :],
            (0, pad_length),
            mode='constant',
            value=0
        )
        return padded_tensor.unsqueeze(0)
    else:
        return waveform[:, :max_length]


def normalize_lufs(audio, sample_rate, target_lufs=-20.0):
    """
    使用 LUFS 標準化音量

    Args:
        audio (Tensor): 音訊波形, shape: (channels, samples)
        sample_rate (int): 取樣率
        target_lufs (float): 目標音量（通常 -23 LUFS）

    Returns:
        Tensor: 標準化後的音訊
    """
    normal = audio.squeeze(0).numpy()
    meter  = pyln.Meter(sample_rate)                # 建立 LUFS 測量器
    lufs   = meter.integrated_loudness(normal)      # 計算音訊 LUFS
    gain   = target_lufs - lufs                     # 計算增益補償
    normal = normal * (10 ** (gain / 20))           # 應用增益
    return torch.from_numpy(normal).unsqueeze(0)  


def normalize_rms(audio, target_dB=-20.0, eps=1e-6):
    """
    調整音訊 RMS 到目標 dB 值
    
    Args:
        audio (Tensor): 音訊波形, shape: (channels, samples)
        target_dB (float): 目標 RMS dB
        eps (float): 避免除 0 問題

    Returns:
        Tensor: 正規化後的音訊
    """
    rms = torch.sqrt(torch.mean(audio ** 2))  # 計算 RMS
    scalar = 10 ** (target_dB / 20) / (rms + eps)  # 計算縮放係數
    return audio * scalar  # 調整音量


class Clotho(torch.utils.data.Dataset):
    def __init__(
            self,
            path: str               = "./datasets/clotho",
            ftype: str              = "development",
            sample_rate: int        = 32000,
            max_sound_length: float = 4.08,
        ):
        self.sample_rate       = sample_rate
        self.max_sound_length  = max_sound_length
        self.wav_path          = os.path.join(path, ftype)
        self.all_wav           = glob.glob(os.path.join(path, ftype, "*.wav"))
        self.csv               = pd.read_csv(
            os.path.join(path, f"clotho_captions_{ftype}.csv"),
            header=0
        )
        self.caption           = 5

    def __getitem__(self, index):
        file_name   = self.csv.iloc[index//self.caption]["file_name"]
        sound1, _   = librosa.load(os.path.join(self.wav_path, file_name), sr=self.sample_rate)
        sound2, _   = librosa.load(random.choice(self.all_wav), sr=self.sample_rate)
        sound1      = torch.from_numpy(sound1).unsqueeze(0)
        sound2      = torch.from_numpy(sound2).unsqueeze(0)
        
        trim_sound1 = pad_waveform(sound1, self.max_sound_length, self.sample_rate)
        trim_sound2 = pad_waveform(sound2, self.max_sound_length, self.sample_rate)
        trim_sound1 = normalize_rms(trim_sound1)
        trim_sound2 = normalize_rms(trim_sound2)

        mix_sound   = trim_sound1 + trim_sound2
        
        caption     = str(self.csv.iloc[index//self.caption][f"caption_{index%self.caption+1}"])
        return mix_sound, caption, trim_sound1

    def __len__(self):
        return len(self.csv) * self.caption


if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader

    dataset    = Clotho()
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4
    )

    for mix, cap, target in dataloader:
        print(mix.shape)
        print(cap)
        print(target.shape)
        break

    