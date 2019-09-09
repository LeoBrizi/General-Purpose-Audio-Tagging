from DataLoader import *
from CNNSpecNetwork import *
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import argparse
from tqdm import tqdm


def main(model_name, spec_version=1):
    model = CNNSpecNetwork()
    model.load_model(model_name)
    loader = DataLoader()
    loader.load_files_labels("./dataset/test_post_competition.csv")
    print(loader.get_label_id_mapping())
    spectrograms, label = loader.load_verified_spectrograms(version=spec_version)
    print("Predict on test set...")
    y_pred = []
    for s in tqdm(spectrograms):
        y = model.predict_single_data(s)
        y_pred.append(y[0])

    y_test_arr = np.array(label)
    y_pred_arr = np.array(y_pred)

    confusionMatrix = confusion_matrix(y_test_arr, y_pred_arr)

    print('Accuracy:')
    print(accuracy_score(y_test_arr, y_pred_arr))
    print('Precision:')
    print(precision_score(y_test_arr, y_pred_arr, average=None))
    print('Recall:')
    print(recall_score(y_test_arr, y_pred_arr, average=None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train the convolutional neural network ")
    parser.add_argument('--model_name', help='name of the model.')
    parser.add_argument(
        '--sv', help='spectrogram version used to train the network', type=int, default=1)
    args = parser.parse_args()

    main(model_name=args.model_name, spec_version=args.sv)
