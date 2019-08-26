class DataManager():

    def __init__(self):
        pass

    @staticmethod
    def conform_dim(spectrogram, frame_size):
        newspectrogram = spectrogram.copy()

        while(newspectrogram.shape[1] < frame_size):
            newspectrogram = np.concatenate(
                (newspectrogram, spectrogram), axis=1)

        newspectrogram = newspectrogram[:, 0:frame_size]
        return newspectrogram

    @staticmethod
    def data_augmentation(data_to_augment):
