import os
from tqdm import tqdm
import numpy as np
import librosa
from Preprocessor import *
from DataManager import *


class DataLoader():

    def __init__(self, sample_rate=32000, dim_to_conform=3000):
        self.class_id_mapping = {'Hi-hat': 0, 'Saxophone': 1, 'Trumpet': 2, 'Glockenspiel': 3, 'Cello': 4, 'Knock': 5, 'Gunshot_or_gunfire': 6, 'Clarinet': 7, 'Computer_keyboard': 8, 'Keys_jangling': 9, 'Snare_drum': 10, 'Writing': 11, 'Laughter': 12, 'Tearing': 13, 'Fart': 14, 'Oboe': 15, 'Flute': 16, 'Cough': 17, 'Telephone': 18, 'Bark': 19, 'Chime': 20, 'Bass_drum': 21,
                                 'Bus': 22, 'Squeak': 23, 'Scissors': 24, 'Harmonica': 25, 'Gong': 26, 'Microwave_oven': 27, 'Burping_or_eructation': 28, 'Double_bass': 29, 'Shatter': 30, 'Fireworks': 31, 'Tambourine': 32, 'Cowbell': 33, 'Electric_piano': 34, 'Meow': 35, 'Drawer_open_or_close': 36, 'Applause': 37, 'Acoustic_guitar': 38, 'Violin_or_fiddle': 39, 'Finger_snapping': 40}

        self.classes_frequency = {}
        self.classes_percent = {}
        self.classes_verified = {}
        self.files = []
        self.labels = []
        self.verified = []

        self.files_loaded = set()

        self.train_csv = False
        self.sample_rate = sample_rate

        self.preprocessor = Preprocessor(dump=True)
        self.dim_to_conform = dim_to_conform

    def load_files_labels(self, file_csv):
        with open(file_csv, 'r') as fp:
            file_list = fp.read()
        fp.close()
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
        tot = len(self.labels)
        for key in self.classes_frequency.keys():
            self.classes_percent[key] = self.classes_frequency[key] / tot
        return self.files, self.labels

    def __load_spectrogram(self, file_name, version):
        if(version == 1):
            data_path = "./dataset/spec/ver1/"
        elif(version == 2):
            data_path = "./dataset/spec/ver2/"
        self.preprocessor.set_spectrogram_path(data_path)
        self.preprocessor.set_version(version)
        spec_file_name = file_name.replace(".wav", ".npy")
        spec_file_name = os.path.join(data_path, spec_file_name)
        try:
            spec = np.load(spec_file_name)
        except FileNotFoundError:
            print(
                file_name, " spectrogram not exist, compute spectrogram from the original file")
            signal = self.__load_audio_signal(file_name)
            spec = self.preprocessor.compute_spectrogram(
                signal, os.path.basename(spec_file_name))
        return spec

    def __load_audio_signal(self, file_name):
        if(self.train_csv):
            audio_path = "./dataset/audio_train/"
        else:
            audio_path = "./dataset/audio_test/"
        audio_file_name = os.path.join(audio_path, file_name)
        signal, sample_rate = librosa.load(
            audio_file_name, sr=self.sample_rate, mono=True)
        signal = self.preprocessor.normalize_and_trim_silence(signal)
        return signal

    def load_verified_spectrograms(self, version):
        if(len(self.files) == 0):
            raise NameError("load the file name list before from csv file")
        labels = []
        spectrograms = []
        print("loading verified spectrograms...")
        for index in tqdm(range(len(self.files))):
            if(not self.verified[index]):
                continue
            file_name = self.files[index]
            spec = self.__load_spectrogram(file_name, version)
            spec = DataManager.conform_dim(spec, self.dim_to_conform, 1)
            spectrograms.append(spec)
            labels.append(self.labels[index])
            self.files_loaded.add(file_name)
        return np.asarray(spectrograms, dtype=np.float32), np.asarray(labels, dtype=np.int32)

    def load_verified_audio_signal(self):
        if(len(self.files) == 0):
            raise NameError("load the file name list before from csv file")
        labels = []
        audio_signals = []
        print("loading audio signals...")
        for index in tqdm(range(len(self.files))):
            if(not self.verified[index]):
                continue
            file_name = self.files[index]
            signal = self.__load_audio_signal(file_name)
            signal = DataManager.conform_dim(signal, self.dim_to_conform, 0)
            audio_signals.append(signal)
            labels.append(self.labels[index])
            self.files_loaded.add(file_name)

        return np.asarray(audio_signals, dtype=np.float32), np.array(labels, dtype=np.int32)

    def __get_file_name_of(self, clas):
        files_of = set()
        for index in range(len(self.files)):
            if(self.labels[index] == clas and self.files[index] in self.files_loaded):
                files_of.add(self.files[index])
        return files_of

    def get_next_spectrograms(self, version, how_many):
        file_per_class = {}
        loaded = 0
        labels = []
        verified = []
        spectrograms = []
        classes_percent = {}
        for key in self.classes_percent.keys():
            classes_percent[self.class_id_mapping[
                key]] = self.classes_percent[key]
        print("loading the next " + str(how_many) + " files...")
        while(loaded < how_many):
            inex = 0
            for index in tqdm(range(len(self.files))):
                if(self.files[index] in self.files_loaded):
                    continue
                if(file_per_class.setdefault(self.labels[index], 0) > classes_percent[self.labels[index]] * how_many):
                    continue
                if(loaded >= how_many):
                    loaded += how_many
                    break
                spec = self.__load_spectrogram(self.files[index], version)
                spec = DataManager.conform_dim(spec, self.dim_to_conform, 1)
                spectrograms.append(spec)
                labels.append(self.labels[index])
                verified.append(self.verified[index])
                self.files_loaded.add(self.files[index])
                loaded += 1
                file_per_class[self.labels[index]] = file_per_class.setdefault(self.labels[
                                                                               index], 0) + 1

            for key in file_per_class.keys():
                if(file_per_class[key] <= (classes_percent[key] * how_many)):
                    files_to_free = self.__get_file_name_of(key)
                    self.files_loaded = self.files_loaded.difference(files_to_free)
                    print(len(self.files_loaded))
        print(len(spectrograms))
        return np.asarray(spectrograms, dtype=np.float32), np.asarray(labels, dtype=np.int32), verified

    def get_next_audio_signals(self, how_many):
        file_per_class = {}
        loaded = 0
        labels = []
        audio_signals = []
        verified = []
        for key in self.classes_percent.keys():
            classes_percent[self.class_id_mapping[
                key]] = self.classes_percent[key]
        print("loading the next " + str(how_many) + " files...")
        while(loaded < how_many):
            for index in tqdm(range(len(self.files))):
                if(self.files[index] in self.files_loaded):
                    continue
                if(file_per_class.setdefault(self.labels[index], 0) > classes_percent[self.labels[index]] * how_many):
                    continue
                if(loaded >= how_many):
                    break
                signal = self.__load_audio_signal(self.files[index], version)
                signal = DataManager.conform_dim(
                    signal, self.dim_to_conform, 1)
                audio_signals.append(signal)
                labels.append(self.labels[index])
                verified.append(self.verified[index])
                self.files_loaded.add(self.files[index])
                loaded += 1
                file_per_class[self.labels[index]] = file_per_class.setdefault(self.labels[
                                                                               index], 0) + 1

            for key in file_per_class.keys():
                if(file_per_class[key] < classes_percent[self.labels[index]] * how_many):
                    files_to_free = self.__get_file_name_of(key)
                    self.files_loaded = self.files_loaded.difference(
                        files_to_free)
        return np.asarray(audio_signals, dtype=np.float32), np.asarray(labels, dtype=np.int32), verified

    def get_label_id_mapping(self):
        return self.class_id_mapping

    def get_label(self):
        return self.class_id_mapping.keys()

    def get_general_statistics(self):
        tot = len(self.labels)
        classes_percent_verified = {}
        for key in self.classes_frequency.keys():
            classes_percent_verified[key] = self.classes_verified[
                key] / self.classes_frequency[key]
        verif_num = self.verified.count(True)
        return self.classes_frequency, self.classes_percent, verif_num, verif_num / tot, self.classes_verified, classes_percent_verified
