import os
from tqdm import tqdm
import numpy as np
import librosa
from Preprocessor import *


class Loader():

    def __init__(self, sample_rate=32000):
        self.class_id_mapping = {'Hi-hat': 0, 'Saxophone': 1, 'Trumpet': 2, 'Glockenspiel': 3, 'Cello': 4, 'Knock': 5, 'Gunshot_or_gunfire': 6, 'Clarinet': 7, 'Computer_keyboard': 8, 'Keys_jangling': 9, 'Snare_drum': 10, 'Writing': 11, 'Laughter': 12, 'Tearing': 13, 'Fart': 14, 'Oboe': 15, 'Flute': 16, 'Cough': 17, 'Telephone': 18, 'Bark': 19, 'Chime': 20, 'Bass_drum': 21,
                                 'Bus': 22, 'Squeak': 23, 'Scissors': 24, 'Harmonica': 25, 'Gong': 26, 'Microwave_oven': 27, 'Burping_or_eructation': 28, 'Double_bass': 29, 'Shatter': 30, 'Fireworks': 31, 'Tambourine': 32, 'Cowbell': 33, 'Electric_piano': 34, 'Meow': 35, 'Drawer_open_or_close': 36, 'Applause': 37, 'Acoustic_guitar': 38, 'Violin_or_fiddle': 39, 'Finger_snapping': 40}

        self.classes_frequency = {}
        self.spec_statistic = {"max": None,
                               "min": None, "average": 0, "variance": 0, "len_hist": {}}
        self.audio_statistics = {"max": None,
                                 "min": None, "average": 0, "variance": 0, "len_hist": {}}
        self.classes_verified = {}
        self.files = []
        self.labels = []
        self.verified = []

        self.spectrograms = []
        self.audio_signals = []

        self.train_csv = False
        self.sample_rate = sample_rate

    def load_files_labels(self, file_csv):
        with open(file_csv, 'r') as fp:
            file_list = fp.read()
        print("load labels and files name from csv...")
        file_list = file_list.split("\n")
        if(file_list[0].split(",")[2].strip() == "manually_verified"):
            self.train_csv = True
        for line in file_list[1:]:
            split_line = line.split(",")
            if(split_line[0] == ''):
                continue

            file_name = split_line[0].strip()
            file_label = split_line[1].strip()
            if(file_label == "None"):
                continue
            if(self.train_csv):
                file_verified = np.bool(split_line[2].strip() == '1')
            else:
                file_verified = np.bool(True)
            self.verified.append(file_verified)

            self.files.append(file_name)

            self.labels.append(self.class_id_mapping[file_label])

            self.classes_frequency[
                file_label] = self.classes_frequency.setdefault(file_label, 0) + 1
            if(file_verified):
                self.classes_verified[
                    file_label] = self.classes_verified.setdefault(file_label, 0) + 1

        print("finish loading csv")
        return self.files, self.labels

    def load_spectrogram(self, version):
        if(len(self.files) == 0):
            raise NameError("load the file name list before from csv file")

        if(version == 1):
            data_path = "./dataset/spec/ver1/"
        elif(version == 2):
            data_path = "./dataset/spec/ver2/"

        preprocessor = Preprocessor(
            spectrogram_path=data_path, version=version, test=False, dump=True)

        if(self.train_csv):
            audio_path = "./dataset/audio_train/"
        else:
            audio_path = "./dataset/audio_test/"

        print("loading spectrograms...")
        for file_name in tqdm(self.files):
            spec_file_name = file_name.replace(".wav", ".npy")
            spec_file_name = os.path.join(data_path, spec_file_name)
            try:
                spec = np.load(spec_file_name)
            except FileNotFoundError:
                print(
                    file_name, " spectrogram not exist, compute spectrogram from the original file")
                audio_file_name = os.path.join(audio_path, file_name)
                signal, sample_rate = librosa.load(
                    audio_file_name, sr=self.sample_rate, mono=True)
                spec = preprocessor.compute_spectrogram(
                    signal, os.path.basename(spec_file_name))
            self.spectrograms.append(spec)
            # compute statistics
            Xk = spec.shape[1]
            if(self.spec_statistic["max"] == None or Xk > self.spec_statistic["max"]):
                self.spec_statistic["max"] = Xk
            if(self.spec_statistic["min"] == None or Xk < self.spec_statistic["min"]):
                self.spec_statistic["min"] = Xk
            k = len(self.spectrograms)
            delta = (Xk - self.spec_statistic["average"]) / k
            self.spec_statistic["average"] = self.spec_statistic[
                "average"] + delta
            self.spec_statistic["variance"] = ((
                (k - 1) * self.spec_statistic["variance"]) / k) + (delta * (Xk - self.spec_statistic["average"]))
            self.spec_statistic["len_hist"][Xk] = self.spec_statistic[
                "len_hist"].setdefault(Xk, 0) + 1

        return self.spectrograms, self.labels

    def load_audio_signal(self):
        if(len(self.files) == 0):
            raise NameError("load the file name list before from csv file")
        preprocessor = Preprocessor()
        if(self.train_csv):
            audio_path = "./dataset/audio_train/"
        else:
            audio_path = "./dataset/audio_test/"

        print("loading audio signals...")
        for file_name in tqdm(self.files):
            audio_file_name = os.path.join(audio_path, file_name)
            signal, sample_rate = librosa.load(
                audio_file_name, sr=self.sample_rate, mono=True)
            signal = preprocessor.normalize_and_trim_silence(signal)
            self.audio_signals.append(signal)
            # compute statistics
            Xk = len(signal)
            if(self.audio_statistics["max"] == None or Xk > self.audio_statistics["max"]):
                self.audio_statistics["max"] = Xk
            if(self.audio_statistics["min"] == None or Xk < self.audio_statistics["min"]):
                self.audio_statistics["min"] = Xk
            k = len(self.audio_signals)
            delta = (Xk - self.audio_statistics["average"]) / k
            self.audio_statistics["average"] = self.audio_statistics[
                "average"] + delta
            self.audio_statistics["variance"] = ((
                (k - 1) * self.audio_statistics["variance"]) / k) + (delta * (Xk - self.audio_statistics["average"]))
            self.audio_statistics["len_hist"][Xk] = self.audio_statistics[
                "len_hist"].setdefault(Xk, 0) + 1

        return self.audio_signals, self.labels

    # with default parameters select all
    def select_spectrogram(self, verified=False, classes=[]):
        if(len(self.spectrograms) == 0):
            raise NameError("load spectrograms list before")
        subset_spec = []
        subset_label = []
        for index in range(len(self.spectrograms)):
            if(verified == self.verified[index]):
                if(classes == [] or self.labels[index] in classes):
                    subset_spec.append(self.spectrograms[index])
                    subset_label.appen(self.labels[index])
        return subset_spec, subset_label

    def select_audio_signal(self, verified=False, classes=[]):
        if(len(self.audio_signals) == 0):
            raise NameError("load audio signals list before")
        subset_audio = []
        subset_label = []
        for index in range(len(self.audio_signals)):
            if(verified == self.verified[index]):
                if(classes == [] or self.labels[index] in classes):
                    subset_audio.append(self.audio_signals[index])
                    subset_label.append(self.labels[index])
        return subset_audio, subset_label

    def get_label_id_mapping(self):
        return self.class_id_mapping

    def get_label(self):
        return self.class_id_mapping.keys()

    def get_general_statistics(self):
        tot = len(self.labels)
        classes_percent = {}
        classes_percent_verified = {}
        for key in self.classes_frequency.keys():
            classes_percent[key] = self.classes_frequency[key] / tot
            classes_percent_verified[key] = self.classes_verified[
                key] / self.classes_frequency[key]
        verif_num = self.verified.count(True)
        return self.classes_frequency, classes_percent, verif_num, verif_num / tot, self.classes_verified, classes_percent_verified

    def get_audio_statistics(self):
        return self.audio_statistics

    def get_spectrogram_statistics(self):
        return self.spec_statistic

    # c as number
    def get_audio_statistics_for_class(self, c):
        class_dict = {"max": None, "min": None,
                      "average": 0, "variance": 0, "len_hist": {}}
        for index in range(len(self.audio_signals)):
            if(self.labels[index] == c):
                Xk = len(self.audio_signals[index])
                if(class_dict["max"] == None or Xk > class_dict["max"]):
                    class_dict["max"] = Xk
                if(class_dict["min"] == None or Xk < class_dict["min"]):
                    class_dict["min"] = Xk
                k = index
                delta = (Xk - class_dict["average"]) / k
                class_dict["average"] = class_dict[
                    "average"] + delta
                class_dict["variance"] = ((
                    (k - 1) * class_dict["variance"]) / k) + (delta * (Xk - class_dict["average"]))
                class_dict["len_hist"][Xk] = class_dict[
                    "len_hist"].setdefault(Xk, 0) + 1
        return class_dict

    def get_spec_statistics_for_class(self, c):
        class_dict = {"max": None, "min": None,
                      "average": 0, "variance": 0, "len_hist": {}}
        for index in range(len(self.spectrograms)):
            if(self.labels[index] == c):
                Xk = self.spectrograms[index].shape[1]
                if(class_dict["max"] == None or Xk > class_dict["max"]):
                    class_dict["max"] = Xk
                if(class_dict["min"] == None or Xk < class_dict["min"]):
                    class_dict["min"] = Xk
                k = index
                delta = (Xk - class_dict["average"]) / k
                class_dict["average"] = class_dict[
                    "average"] + delta
                class_dict["variance"] = ((
                    (k - 1) * class_dict["variance"]) / k) + (delta * (Xk - class_dict["average"]))
                class_dict["len_hist"][Xk] = class_dict[
                    "len_hist"].setdefault(Xk, 0) + 1
        return class_dict
