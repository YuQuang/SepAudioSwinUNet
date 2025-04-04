from models.swin_unet_lass import SwinUnetLASS

if __name__ == "__main__":
    import torch
    import librosa
    from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

    audio, _ = librosa.load("./datasets/Clotho/development/1.wav", sr=32000)
    audio = torch.from_numpy(audio[:130560]).view(1, 1, -1).to("cuda")
    model = SwinUnetLASS().to("cuda")
    out = model(audio)
    loss = scale_invariant_signal_noise_ratio(audio, out)
    print(loss)