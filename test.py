from Loader import *
from DataManager import *
from CNNSpecNetwork import *
import time
from DataLoader import *

loader = Loader()
loader.load_files_labels("./dataset/train.csv")

spec, label = loader.load_spectrogram(version=1)
print("---------------------------------------------------------------------------")
print(loader.get_general_statistics())
print("---------------------------------------------------------------------------")
print(loader.get_spectrogram_statistics())
spec = DataManager.conform_vector(spec, 1000, 1)
spec = np.asarray(spec, dtype=np.float64)

print("---------------------------------------------")
print(spec.shape)
print("---------------------------------------------")
print(label)

time.sleep(10)
