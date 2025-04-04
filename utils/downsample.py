import glob
import librosa
import soundfile as sf

DOWNSAMPLE_PATH = "./proccesed_dir/"

if __name__ == "__main__":

    y, s = librosa.load('./datasets/Clotho/development/140704_rain.wav', sr=32000)
    print(y.shape[0] / 32000)
    sf.write('./test.wav', y, 32000, subtype='PCM_24')

    