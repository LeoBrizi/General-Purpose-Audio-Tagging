from Loader import *
from DataManager import *

loader = Loader()
loader.load_files_labels("./dataset/train.csv")

audio, label = loader.load_spectrogram(version=1)
print("---------------------------------------------------------------------------")
map = loader.get_label_id_mapping()
print("---------------------------------------------------------------------------")
print(loader.get_general_statistics())
print("---------------------------------------------------------------------------")
print(loader.get_spectrogram_statistics())

print("------------------------------FART----------------------------------------------")
print(loader.get_spec_statistics_for_class(map["Fart"]))
