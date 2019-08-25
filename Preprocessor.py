import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import matplotlib.pyplot as plt

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.audio.filters import LogFilterbank
from madmom.processors import SequentialProcessor, Processor


class Preprocessor():

    def __init__(self, spectrogram_path, version=1, test=False, dump=False, preprocessing=True, sample_rate=32000, frame_size=3000, silence_threshold=40):
        if(version != 1 and version != 2):
            raise NameError("version must be 1 or 2")
        if(dump and spectrogram_path == None):
            raise NameError(
                "if you want to dump the spectrograms, You have to specifie a directory where store dumps")
        if(dump):
            if not os.path.exists(spectrogram_path):
                os.makedirs(spectrogram_path)
        self.version = version
        self.spectrogram_path = spectrogram_path
        self.sample_rate = sample_rate
        self.preprocessing = preprocessing
        self.test = test
        self.dump = dump
        self.silence_threshold = silence_threshold
        self.frame_size = frame_size

        sig_proc = SignalProcessor(
            num_channels=1, sample_rate=self.sample_rate, norm=True)
        fsig_proc = FramedSignalProcessor(
            frame_size=1024, hop_size=128, origin='future')
        spec_proc = SpectrogramProcessor(frame_size=1024)
        filt_proc = LogarithmicFilteredSpectrogramProcessor(
            filterbank=LogFilterbank, num_bands=26, fmin=20, fmax=14000)
        processor_pipeline = [sig_proc, fsig_proc, spec_proc, filt_proc]
        self.processor_version2 = SequentialProcessor(processor_pipeline)

    def __processV1(self, signal, fft_window_size, hop_length, log_spectrogram, n_mels, fmax):
        # compute stft
        stft = librosa.stft(signal, n_fft=fft_window_size, hop_length=hop_length, win_length=None, window='hann', center=True,
                            pad_mode='reflect')
        # keep only magnitude
        stft = np.abs(stft)
        # spectrogram weighting
        if log_spectrogram:
            stft = np.log10(stft + 1)
        else:
            freqs = librosa.core.fft_frequencies(
                sr=self.sample_rate, n_fft=fft_window_size)
            stft = librosa.perceptual_weighting(
                stft**2, freqs, ref=1.0, amin=1e-10, top_db=99.0)
        # apply mel filterbank
        spectrogram = librosa.feature.melspectrogram(
            S=stft, sr=self.sample_rate, n_mels=n_mels, fmax=fmax)

        spectrogram = np.asarray(spectrogram)
        return spectrogram

    def __processV2(self, signal):
        spectrogram = self.processor_version2.process(signal)
        return spectrogram

    def process(self, file_path):
        signal, self.sample_rate = librosa.load(
            file_path, sr=self.sample_rate, mono=True)

        if(self.preprocessing):
            signal = librosa.util.normalize(signal, norm=100)
            # trim silence at beginning and end and normalize to -0.1
            signal, index = librosa.effects.trim(
                signal, top_db=self.silence_threshold)

        if(self.version == 1):
            spectrogram = self.__processV1(
                signal, fft_window_size=1024, hop_length=192, log_spectrogram=False, n_mels=128, fmax=None)
        else:
            spectrogram = self.__processV2(signal)
            spectrogram = np.swapaxes(spectrogram, 0, 1)

        # plot spectrogram
        if self.test:
            print("Spectrogram Shape:", spectrogram.shape)
            plt.figure("General-Purpose ")
            plt.clf()
            plt.subplots_adjust(right=0.98, left=0.1, bottom=0.1, top=0.99)
            plt.imshow(spectrogram, origin="lower",
                       interpolation="nearest", cmap="viridis")
            plt.xlabel("%d frames" % spectrogram.shape[1])
            plt.ylabel("%d bins" % spectrogram.shape[0])
            plt.colorbar()
            plt.show()
            plt.show(block=True)

        if self.dump:
            # save spectrograms
            spec_file = os.path.join(self.spectrogram_path, os.path.basename(
                file_path).replace(".wav", ".npy"))
            np.save(spec_file, spectrogram)

        return spectrogram
