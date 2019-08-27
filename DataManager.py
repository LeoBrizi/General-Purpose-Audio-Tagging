import numpy as np
import librosa
import tqdm


class DataManager():

    def __init__(self, preprocessor, loader):
        self.data_augmentation_dir = "./dataset/aug/"
        self.augmented_spec_v1_dir = "./dataset/aug/spec/ver1/"
        self.augmented_spec_v2_dir = "./dataset/aug/spec/ver2/"
        self.preprocessor = preprocessor
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

    def audio_signals_augmentation(self, class_to_augment, many):
        classes = [class_to_augment]
        
        try:
            original_audio = loader.select_audio_signal(verified=True, classes)
        except Exception as e:
            loader.load_audio_signal()
            original_audio = loader.select_audio_signal(verified=True, classes)
        audio_signals_aug = []
        for orig_audio_sig in tqdm(original_audio):

            #change pitch and speed
            y_pitch_speed = orig_audio_sig.copy()
            # you can change low and high here
            length_change = np.random.uniform(low=0.8, high = 1)
            speed_fac = 1.0  / length_change
            print("resample length_change = ",length_change)
            tmp = np.interp(np.arange(0, len(y_pitch_speed),speed_fac), np.arange(0, len(y_pitch_speed)), y_pitch_speed)
            min_len = min(y_pitch_speed.shape[0], tmp.shape[0])
            y_pitch_speed *= 0
            y_pitch_speed[0:min_len] = tmp[0:min_len]
            librosa.write




    def spectrograms_augmentation(self, class_to_augment, many):

