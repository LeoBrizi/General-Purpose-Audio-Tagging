import os
import glob
import argparse
from tqdm import tqdm
import librosa
from Preprocessor import *

if __name__ == "__main__":
    # add argument parser
    parser = argparse.ArgumentParser(
        description='Compute spectrograms for training and testing audio files.')
    parser.add_argument('--audio_path', help='path to audio files.')
    parser.add_argument('--spec_path', help='path where store spectrograms.')
    parser.add_argument(
        '--test', help='show spectrogram plots.', action='store_true')
    parser.add_argument(
        '--dump', help='dump spectrograms on file.', action='store_true')
    parser.add_argument(
        '--spec_version', help='spectrogram version to compute (1 or 2).', type=int, default=1)
    parser.add_argument(
        '--no_preprocessing', help='normalize to -0.1 and trim silence at bebinning and end', action='store_true')
    args = parser.parse_args()

    # get list of audio files
    file_list = glob.glob(os.path.join(args.audio_path, "*.wav"))

    # create preprocessor
    preprocessor = Preprocessor(spectrogram_path=args.spec_path, version=args.spec_version,
                                test=args.test, dump=args.dump, preprocessing=(not args.no_preprocessing))
    for file in tqdm(file_list):
        signal, sample_rate = librosa.load(
            file, sr=32000, mono=True)
        preprocessor.compute_spectrogram(signal)
