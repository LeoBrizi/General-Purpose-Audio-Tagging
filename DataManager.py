import numpy as np
import librosa
from tqdm import tqdm
import os
import shutil
from Loader import *
from Preprocessor import *


class DataManager():

    def __init__(self, loader):
        self.data_augmentation_dir = "./dataset/aug/"
        self.augmented_spec_v1_dir = "./dataset/aug/spec/ver1/"
        self.augmented_spec_v2_dir = "./dataset/aug/spec/ver2/"
        self.loader = loader

    @staticmethod
    def conform_dim(data, size, axis_to_conform):
        new_data = data.copy()

        while(new_data.shape[axis_to_conform] < size):
            new_data = np.concatenate(
                (new_data, data), axis=axis_to_conform)

        if(axis_to_conform == 0):
            new_data = new_data[0:size]

        elif(axis_to_conform == 1):
            new_data = new_data[:, 0:size]

        return new_data

    @staticmethod
    def change_pitch_and_speed(orig_audio_sig):
        y_pitch_speed = orig_audio_sig.copy()
        length_change = np.random.uniform(low=0.8, high=1)
        speed_fac = 1.0 / length_change
        tmp = np.interp(np.arange(0, len(y_pitch_speed), speed_fac), np.arange(
            0, len(y_pitch_speed)), y_pitch_speed)
        min_len = min(y_pitch_speed.shape[0], tmp.shape[0])
        y_pitch_speed *= 0
        y_pitch_speed[0:min_len] = tmp[0:min_len]
        return y_pitch_speed

    @staticmethod
    def change_pitch(orig_audio_sig):
        y_pitch = orig_audio_sig.copy()
        bins_per_octave = 12
        pitch_pm = 2
        pitch_change = pitch_pm * 2 * (np.random.uniform())
        y_pitch = librosa.effects.pitch_shift(y_pitch.astype(
            'float64'), loader.sample_rate, n_steps=pitch_change, bins_per_octave=bins_per_octave)
        return y_pitch

    @staticmethod
    def change_speed(orig_audio_sig):
        y_speed = orig_audio_sig.copy()
        speed_change = np.random.uniform(low=0.9, high=1.1)
        tmp = librosa.effects.time_stretch(
            y_speed.astype('float64'), speed_change)
        minlen = min(y_speed.shape[0], tmp.shape[0])
        y_speed *= 0
        y_speed[0:minlen] = tmp[0:minlen]
        return y_speed

    @staticmethod
    def value_augmentation(orig_audio_sig):
        y_aug = orig_audio_sig.copy()
        dyn_change = np.random.uniform(low=1.5, high=3)
        y_aug = y_aug * dyn_change
        return y_aug

    @staticmethod
    def add_distribution_noise(orig_audio_sig):
        y_noise = orig_audio_sig.copy()
        noise_amp = 0.005 * np.random.uniform() * np.amax(y_noise)
        y_noise = y_noise.astype(
            'float64') + noise_amp * np.random.normal(size=y_noise.shape[0])
        return y_noise

    @staticmethod
    def random_shifting(orig_audio_sig):
        y_shift = orig_audio_sig.copy()
        timeshift_fac = 0.2 * 2 * \
            (np.random.uniform() - 0.5)  # up to 20% of length
        start = int(y_shift.shape[0] * timeshift_fac)
        if (start > 0):
            y_shift = np.pad(y_shift, (start, 0), mode='constant')[
                0:y_shift.shape[0]]
        else:
            y_shift = np.pad(y_shift, (0, -start),
                             mode='constant')[0:y_shift.shape[0]]
        return y_shift

    @staticmethod
    def streching(orig_audio_sig):
        input_length = orig_audio_sig.shape[0]
        streching = orig_audio_sig.copy()
        streching = librosa.effects.time_stretch(
            streching.astype('float'), 1.1)
        if len(streching) > input_length:
            streching = streching[:input_length]
        else:
            streching = np.pad(
                streching, (0, max(0, input_length - len(streching))), "constant")
        return streching

    def audio_signals_augmentation(self, class_to_augment, many, methods=[]):
        audio_signals_aug = []
        aug_dir = self.data_augmentation_dir + class_to_augment + "/"
        if not os.path.exists(aug_dir):
            os.makedirs(aug_dir)
        files = os.listdir(aug_dir)
        if(len(files) >= many):
            for file in files:
                signal, sample_rate = librosa.load(
                    file, sr=loader.sample_rate, mono=True)
                audio_signals_aug.append(signal)
                if(len(audio_signals_aug) >= many):
                    break
            return np.asarray(audio_signals_aug, dtype=float64)

        preprocessor = Preprocessor(None, version=1)
        shutil.rmtree(aug_dir)
        os.makedirs(aug_dir)
        classes = [class_to_augment]
        try:
            original_audio = loader.select_audio_signal(
                verified=True, classes=classes)
        except Exception as e:
            loader.load_audio_signal()
            original_audio = loader.select_audio_signal(
                verified=True, classes=classes)
        name = 0
        print("creating new audio files...")
        for orig_audio_sig in tqdm(original_audio):

            # change pitch and speed
            if(method == [] or "pitch_speed" in methods):
                y_pitch_speed = change_pitch_and_speed(original_audio)
                y_pitch_speed = preprocessor.normalize_and_trim_silence(
                    y_pitch_speed)
                audio_signals_aug.append(y_pitch_speed)
                librosa.output.write_wav(
                    aug_dir + str(name) + "_p_s.wav", y_pitch_speed, loader.sample_rate)

            # change pitch only
            if(method == [] or "pitch" in methods):
                y_pitch = change_pitch(original_audio)
                y_pitch = preprocessor.normalize_and_trim_silence(y_pitch)
                audio_signals_aug.append(y_pitch)
                librosa.output.write_wav(
                    aug_dir + str(name) + "_p.wav", y_pitch, loader.sample_rate)

            # change speed only
            if(method == [] or "speed" in methods):
                y_speed = change_speed(original_audio)
                y_speed = preprocessor.normalize_and_trim_silence(y_speed)
                audio_signals_aug.append(y_speed)
                librosa.output.write_wav(
                    aug_dir + str(name) + "_s.wav", y_speed, loader.sample_rate)

            # value augmentation
            if(method == [] or "aug" in methods):
                y_aug = value_augmentation(original_audio)
                y_aug = preprocessor.normalize_and_trim_silence(y_aug)
                audio_signals_aug.append(y_aug)
                librosa.output.write_wav(
                    aug_dir + str(name) + "_aug.wav", y_aug, loader.sample_rate)

            # add distribution noise
            if(method == [] or "noise" in methods):
                y_noise = add_distribution_noise(original_audio)
                y_noise = preprocessor.normalize_and_trim_silence(y_noise)
                audio_signals_aug.append(y_noise)
                librosa.output.write_wav(
                    aug_dir + str(name) + "_noise.wav", y_noise, loader.sample_rate)

            # random shifting
            if(method == [] or "shift" in methods):
                y_shift = random_shifting(original_audio)
                y_shift = preprocessor.normalize_and_trim_silence(y_shift)
                audio_signals_aug.append(y_shift)
                librosa.output.write_wav(
                    aug_dir + str(name) + "_shift.wav", y_shift, loader.sample_rate)

            # streching
            if(method == [] or "strech" in methods):
                streching = streching(original_audio)
                streching = preprocessor.normalize_and_trim_silence(streching)
                audio_signals_aug.append(streching)
                librosa.output.write_wav(
                    aug_dir + str(name) + "_str.wav", streching, loader.sample_rate)

            name += 1
            if(len(audio_signals_aug) >= many):
                break
        return audio_signals_aug

    def spectrograms_augmentation(self, class_to_augment, many, version):
        spec_aug = []
        if(version == 1):
            aug_dir = self.augmented_spec_v1_dir + class_to_augment + "/"
        elif(version == 2):
            aug_dir = self.augmented_spec_v2_dir + class_to_augment + "/"
        if not os.path.exists(aug_dir):
            os.makedirs(aug_dir)
        files = os.listdir(aug_dir)
        if(len(files) >= many):
            for file in files:
                spec = np.load(file)
                spec_aug.append(spec)
                if(len(spec_aug) >= many):
                    break
            return np.asarray(spec_aug, dtype=float64)
        preprocessor = Preprocessor(aug_dir, version=version, dump=True)

        shutil.rmtree(aug_dir)
        os.makedirs(aug_dir)
        audio_signals_aug = self.audio_signals_augmentation(
            class_to_augment, many)
        print("creating new spectrograms...")
        name = 0
        for sig_aug in tqdm(audio_signals_aug):
            spec = preprocessor.compute_spectrogram(
                sig_aug, str(name) + ".npy")
            spec_aug.append(spec)
            name += 1
            if(len(spec_aug) >= many):
                break
        return spec_aug
