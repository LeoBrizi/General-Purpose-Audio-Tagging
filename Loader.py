import os
import tqdm
import numpy as np
import librosa
from Preprocessor import *


class Loader():

    def __init__(self):
        self.class_id_mapping = {}

        self.classes_frequency = {}
        self.spec_statistic = {"max": None,
                               "min": None, "average": 0, "variance": 0}
        self.audio_statistics = {"max": None,
                                 "min": None, "average": 0, "variance": 0}
        self.files = []
        self.labels = []
        self.verified = []

        self.spectrograms = []
        self.audio_signals = []

        self.train_csv = False

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
                file_verified = split_line[2].strip()
                self.verified.append(np.bool(file_verified == '1'))
            else:
                self.verified.append(np.bool(True))

            self.files.append(file_name)

            self.class_id_mapping.setdefault(
                file_label, len(self.class_id_mapping.keys()))
            self.labels.append(self.class_id_mapping[file_label])
            self.classes_frequency[file_label]
                = self.classes_frequency.setdefault(file_label, 0) + 1

        print("finish loading csv")
        return np.asarray(self.files, dtype=np.string_), np.asarray(self.labels, dtype=np.int32)

    def load_spectrogram(self, version):
        if(len(self.files) == 0):
            raise NameError("load the file name list before from csv file")

        if(version == 1):
            data_path = "./dataset/spec/ver1/"
            preprocessor = Preprocessor(spectrogram_path=data_path, version=1,
                                        test=False, dump=True)
        elif(version == 2):
            data_path = "./dataset/spec/ver2/"
            preprocessor = Preprocessor(spectrogram_path=data_path, version=2,
                                        test=False, dump=True)
        else:
            raise NameError("version must be 1 or 2")

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
                    audio_file_name, sr=32000, mono=True)
                spec = preprocessor.compute_spectrogram(signal)
            self.spectrograms.append(spec)
            # compute statistics
            Xk = len(spec.shape[1])
            if(self.spec_statistic["max"] == None or Xk > self.spec_statistic["max"]):
                self.spec_statistic["max"] = Xk
            if(self.spec_statistic["min"] == None or Xk < self.spec_statistic["min"]):
                self.spec_statistic["min"] = Xk
            k = len(self.spectrograms)
            delta = Xk - self.spec_statistic["average"] / k
            self.spec_statistic["average"] = self.spec_statistic[
                "average"] + delta
            self.spec_statistic["variance"] = ((k - 1) * self.spec_statistic["variance"]) / k +
                delta * (Xk - self.spec_statistic["average"])

        return np.asarray(self.spectrograms), np.asarray(self.labels, dtype=np.int32)

    def load_audio_signal(self):
        preprocessor = Preprocessor()
        if(len(self.files) == 0):
            raise NameError("load the file name list before from csv file")
        if(self.train_csv):
            audio_path = "./dataset/audio_train/"
        else:
            audio_path = "./dataset/audio_test/"

        print("loading audio signals...")
        for file_name in tqdm(self.files):
            audio_file_name = os.path.join(audio_path, file_name)
            signal, sample_rate = librosa.load(
                audio_file_name, sr=32000, mono=True)
            signal = preprocessor.normalize_and_trim_silence(signal)
            self.audio_signals.append(signal)
            # compute statistics
            Xk = len(signal)
            if(self.audio_statistic["max"] == None or Xk > self.audio_statistic["max"]):
                self.audio_statistic["max"] = Xk
            if(self.audio_statistic["min"] == None or Xk < self.audio_statistic["min"]):
                self.audio_statistic["min"] = Xk
            k = len(self.audio_signals)
            delta = Xk - self.audio_statistic["average"] / k
            self.audio_statistic["average"] = self.audio_statistic[
                "average"] + delta
            self.audio_statistic["variance"] = ((k - 1) * self.audio_statistic["variance"]) / k +
                delta * (Xk - self.audio_statistic["average"])

        return np.asarray(self.audio_signals), np.asarray(self.labels, dtype=np.int32)

    # with default parameters select all
    def select_spectrogram(self, only_verified=False, classes=[]):
        subset_spec = []
        subset_label = []
        for index in range(0, len(self.spectrograms)):
            if(not only_verified or (only_verified and self.verified[index])):
                if(classes == [] or self.labels[index] in classes):
                    subset_spec.append(self.spectrograms[index])
                    subset_label.appen(self.labels[index])
        return np.asarray(subset_spec), np.asarray(subset_label, dtype=np.int32)

    def select_audio_signal(self, only_verified=False, classes=[]):
        subset_audio = []
        subset_label = []
        for index in range(0, len(self.audio_signals)):
            if(not only_verified or (only_verified and self.verified[index])):
                if(classes == [] or self.labels[index] in classes):
                    subset_audio.append(self.audio_signals[index])
                    subset_label.append(self.labels[index])
        return np.asarray(subset_audio), np.asarray(subset_label, dtype=np.int32)

    def get_label_id_mapping(self):
        return self.class_id_mapping

    def get_label(self):
        return self.class_id_mapping.key()

    def get_general_statistics(self):
        tot = len(self.labels)
        classes_percent = {}
        for key in self.classes_frequency.key():
            classes_percent[key] = self.classes_frequency[key] / tot
        verif_num = self.verified.count(True)
        return self.classes_frequency, verif_num, classes_percent, verif_num / tot

    def get_audio_statistics(self):
        return self.audio_statistic

    def get_spectrogram_statistics(self):
        return self.spec_statistic

    # c as number
    def get_audio_statistics_for_class(self, c):
        class_dict = {"max": None, "min": None, "average": 0, "variance": 0}
        for index in range(0, len(self.audio_signals)):
            if(self.labels[index] == c):
                Xk = len(self.audio_signals[index])
                if(class_dict["max"] == None or Xk > class_dict["max"]):
                    class_dict["max"] = Xk
                if(class_dict["min"] == None or Xk < class_dict["min"]):
                    class_dict["min"] = Xk
                k = index
                delta = Xk - class_dict["average"] / k
                class_dict["average"] = class_dict[
                    "average"] + delta
                class_dict["variance"] = ((k - 1) * class_dict["variance"]) / k +
                    delta * (Xk - class_dict["average"])
        return class_dict

    def get_spec_statistics_for_class(self, c):
        class_dict = {"max": None, "min": None, "average": 0, "variance": 0}
        for index in range(0, len(self.spectrograms)):
            if(self.labels[index] == c):
                Xk = len(self.spectrograms[index])
                if(class_dict["max"] == None or Xk > class_dict["max"]):
                    class_dict["max"] = Xk
                if(class_dict["min"] == None or Xk < class_dict["min"]):
                    class_dict["min"] = Xk
                k = index
                delta = Xk - class_dict["average"] / k
                class_dict["average"] = class_dict[
                    "average"] + delta
                class_dict["variance"] = ((k - 1) * class_dict["variance"]) / k +
                    delta * (Xk - class_dict["average"])
        return class_dict

    def show_statistics(self):